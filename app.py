from flask import Flask, request, jsonify
import tempfile, os, uuid, datetime, json

# Import your classifier and config
from TransformerClassify import classify_transformer, CFG

app = Flask(__name__)

BOX_COLORS = {
    "FAULT": "255,0,0",
    "POTENTIAL": "255,255,0",
    "NORMAL": "0,255,0",
}

def tag_for(label: str) -> str:
    # Map your classifier's label to the tag used by the frontend
    if label.lower().startswith("fault"):
        return "FAULT"
    if label.lower().startswith("potential"):
        return "POTENTIAL"
    return "NORMAL"

def dominant_color_for(tag: str) -> str:
    return {"FAULT": "red", "POTENTIAL": "yellow", "NORMAL": "green"}[tag]

def rgb_array_for(tag: str):
    s = BOX_COLORS[tag]
    r, g, b = map(int, s.split(","))
    return [r, g, b]

def elongated_from_box(w: int, h: int, ar_thresh: float) -> bool:
    # Use same heuristic as your script
    return (max(w, h) / max(1, min(w, h))) >= ar_thresh

def confidence_from_metrics(res: dict, tag: str) -> float:
    """
    Heuristic confidence using your returned metrics:
      - Fault: red area ratio, histogram distance, red/bg increase
      - Potential: yellow area ratio, overall hot ratio
      - Normal: high confidence if no boxes
    """
    # Safe access
    red_ratio = float(res.get("area_red", 0.0))
    yel_ratio = float(res.get("area_yellow", 0.0))
    hot_ratio = float(res.get("area_hot", 0.0))
    hist_dist = float(res.get("hist_dist", 0.0))
    red_bg_b  = float(res.get("red_bg_baseline", 0.0))
    red_bg_c  = float(res.get("red_bg_current", 0.0))
    inc = max(0.0, red_bg_c - red_bg_b)

    if tag == "FAULT":
        # Blend of signals; capped nicely
        s1 = min(1.0, red_ratio / max(1e-9, CFG["fault_red_ratio"]))
        s2 = min(1.0, hist_dist / max(1e-9, CFG["hist_distance_min"]))
        # Normalize increase vs configured min increase (scaled by baseline)
        denom = CFG["red_bg_ratio_min_increase"] * max(red_bg_b, 1.0)
        s3 = min(1.0, inc / max(1e-9, denom))
        conf = 0.55 + 0.45 * (0.5 * s1 + 0.3 * s2 + 0.2 * s3)
        return round(min(0.99, conf), 2)

    if tag == "POTENTIAL":
        s1 = min(1.0, yel_ratio / max(1e-9, CFG["potential_yellow_ratio"]))
        s2 = min(1.0, hot_ratio / max(1e-9, CFG["fullwire_hot_fraction"]))
        conf = 0.5 + 0.4 * (0.7 * s1 + 0.3 * s2)
        return round(min(0.95, conf), 2)

    # NORMAL
    return 0.9 if not res.get("boxes") else 0.7

def build_regions_from_boxes(res: dict, tag: str):
    """
    Turn the list of (x,y,w,h) from classify_transformer into the detailed regions
    your frontend expects. One region per box.
    """
    regions = []
    boxes = res.get("boxes", [])
    # If classify_transformer returned tuples, make sure they're plain lists
    for i, box in enumerate(boxes, start=1):
        x, y, w, h = map(int, box)
        cx = x + w / 2.0
        cy = y + h / 2.0
        aspect = round(w / max(1.0, h), 3)
        elongated = elongated_from_box(w, h, CFG["elongated_aspect_ratio"])

        # Use a simple mapping; you can refine if you later compute per-region color
        dom_color = dominant_color_for(tag)
        color_rgb = rgb_array_for(tag)

        # Simple type mapping: FAULT => Hotspot, POTENTIAL => Warmspot, NORMAL => "" (unused)
        rtype = "Hotspot" if tag == "FAULT" else ("Warmspot" if tag == "POTENTIAL" else "")

        regions.append({
            "dbId": i,                # placeholder if you don't have DB ids yet
            "regionId": i,            # unique per response
            "type": rtype,
            "dominantColor": dom_color,
            "colorRgb": color_rgb,
            "boundingBox": {
                "x": x, "y": y, "width": w, "height": h,
                "areaPx": int(w * h)
            },
            "centroid": {"x": int(round(cx)), "y": int(round(cy))},
            "aspectRatio": float(aspect),
            "elongated": bool(elongated),
            "connectedToWire": bool(elongated),   # proxy: treat elongated as wired-like
            "tag": tag,
            # Fill with overall confidence; if you later compute per-region scores, plug them here
            "confidence": confidence_from_metrics(res, tag)
        })
    return regions

@app.route("/detect-anomalies", methods=['POST'])
def api_classify():
    """
    POST /api/classify
    multipart/form-data with three files:
        - baseline:   baseline image file
        - candidate:  current image file
        - config:     optional JSON config file to override defaults
    Optional query/string:
        - returnAnnotated (0/1): include path string to annotated image in payload (debug)
    """
    if "baseline" not in request.files or "candidate" not in request.files:
        error_msg = f"Upload two images with keys 'baseline' and 'candidate'. Received: {list(request.files.keys())}"
        return jsonify({"error": error_msg}), 400

    f_base = request.files["baseline"]
    f_curr = request.files["candidate"]
    f_config = request.files.get("config")  # Optional config file

    return_annot = str(request.args.get("returnAnnotated", "0")).strip() in ("1", "true", "True")

    # Temporary workspace per request
    with tempfile.TemporaryDirectory() as td:
        # Preserve extensions if possible
        b_ext = os.path.splitext(f_base.filename or "")[1] or ".png"
        c_ext = os.path.splitext(f_curr.filename or "")[1] or ".png"

        b_path = os.path.join(td, f"baseline{b_ext}")
        c_path = os.path.join(td, f"candidate{c_ext}")
        out_path = os.path.join(td, "candidate_annotated.png")

        f_base.save(b_path)
        f_curr.save(c_path)

        # Handle optional config override
        runtime_cfg = CFG.copy()  # Start with default config
        
        if f_config:
            try:
                config_data = json.load(f_config.stream)
                print(f"[DEBUG] Received config data: {json.dumps(config_data, indent=2)}")
        
                runtime_cfg.update(config_data)  # Merge/override with provided config
                print(f"[DEBUG] Runtime config after merge: {json.dumps(runtime_cfg, indent=2)}")
            except json.JSONDecodeError as e:
                return jsonify({"error": f"Invalid JSON in config file: {str(e)}"}), 400
            except Exception as e:
                return jsonify({"error": f"Error reading config: {str(e)}"}), 400

        # Run your existing classifier with runtime config (it writes annotated image to out_path)
        res = classify_transformer(b_path, c_path, out_path, runtime_cfg)

        # Build response
        tag = tag_for(res.get("classification", "Normal"))
        regions = build_regions_from_boxes(res, tag)

        payload = {
            "fault_regions": regions if tag != "NORMAL" else [],
            "display_metadata": {
                "id": 1,  # placeholder; put your own id source if needed
                "boxColors": BOX_COLORS,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        }

        # Optional: include annotated image path for debugging (not in your strict schema)
        if return_annot:
            payload["annotated_image"] = res.get("output_image", "")

        return jsonify(payload), 200

def calculate_iou(box1: dict, box2: dict) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_min = box1['x']
    y1_min = box1['y']
    x1_max = x1_min + box1['width']
    y1_max = y1_min + box1['height']
    
    x2_min = box2['x']
    y2_min = box2['y']
    x2_max = x2_min + box2['width']
    y2_max = y2_min + box2['height']
    
    # Calculate intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_regions(ground_truth: list, predictions: list, iou_threshold: float = 0.3) -> dict:
    """
    Match predicted regions to ground truth regions using IoU.
    Returns: {
        'true_positives': [(gt, pred), ...],
        'false_positives': [pred, ...],
        'false_negatives': [gt, ...]
    }
    """
    matched_gt = set()
    matched_pred = set()
    true_positives = []
    
    # Find matches
    for i, gt in enumerate(ground_truth):
        best_iou = 0.0
        best_pred_idx = -1
        
        for j, pred in enumerate(predictions):
            if j in matched_pred:
                continue
            
            gt_box = gt.get('boundingBox', {})
            pred_box = pred.get('boundingBox', {})
            
            if not gt_box or not pred_box:
                continue
                
            iou = calculate_iou(gt_box, pred_box)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = j
        
        if best_pred_idx >= 0:
            matched_gt.add(i)
            matched_pred.add(best_pred_idx)
            true_positives.append((ground_truth[i], predictions[best_pred_idx]))
    
    # Identify false positives and false negatives
    false_positives = [pred for i, pred in enumerate(predictions) if i not in matched_pred]
    false_negatives = [gt for i, gt in enumerate(ground_truth) if i not in matched_gt]
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def update_config_parameters(baseline_image_path: str, 
                             maintenance_image_path: str, 
                             stored_config: dict, 
                             anomaly_results: dict,
                             original_anomaly_results: dict) -> tuple:
    """
    Update configuration parameters by comparing ground truth (edited) with classifier output.
    
    Algorithm:
    1. Match ground truth regions with predicted regions using IoU
    2. Analyze false positives (over-detection) -> increase thresholds slightly
    3. Analyze false negatives (missed detection) -> decrease thresholds slightly
    4. Make conservative adjustments (5-15% change per iteration)
    5. Focus on key detection parameters: fault_red_ratio, hist_distance_min, 
       potential_yellow_ratio, min_blob_area_px, min_cluster_area_px
    
    Args:
        baseline_image_path: Path to baseline image
        maintenance_image_path: Path to maintenance image
        stored_config: Current configuration
        anomaly_results: Ground truth (user-edited) results
        original_anomaly_results: Classifier's original predictions
    
    Returns:
        tuple: (updated_config dict, edited_fault_count int, original_fault_count int)
    """
    import copy
    
    # Extract fault regions
    ground_truth_regions = anomaly_results.get("fault_regions", [])
    predicted_regions = original_anomaly_results.get("fault_regions", [])
    
    edited_fault_count = len(ground_truth_regions)
    original_fault_count = len(predicted_regions)
    
    # Start with a copy of the stored config
    updated_config = copy.deepcopy(stored_config)
    
    # If no differences, return unchanged config
    if edited_fault_count == original_fault_count == 0:
        print("[CONFIG-TUNING] No regions in either set - no tuning needed")
        return updated_config, edited_fault_count, original_fault_count
    
    # Match regions to identify TP, FP, FN
    matches = match_regions(ground_truth_regions, predicted_regions)
    
    tp_count = len(matches['true_positives'])
    fp_count = len(matches['false_positives'])
    fn_count = len(matches['false_negatives'])
    
    print(f"\n[CONFIG-TUNING] Analysis:")
    print(f"  Ground Truth Regions: {edited_fault_count}")
    print(f"  Predicted Regions: {original_fault_count}")
    print(f"  True Positives: {tp_count}")
    print(f"  False Positives: {fp_count} (over-detection)")
    print(f"  False Negatives: {fn_count} (missed detection)")
    
    # Conservative adjustment factors
    INCREASE_FACTOR = 1.08  # Increase by 8% to reduce false positives
    DECREASE_FACTOR = 0.92  # Decrease by 8% to reduce false negatives
    AREA_INCREASE = 1.10    # 10% for area thresholds
    AREA_DECREASE = 0.90
    
    # Determine adjustment strategy
    adjustments_made = []
    
    # Case 1: Too many false positives (over-detection)
    # Increase thresholds to make detection more strict
    if fp_count > 0 and fp_count > fn_count:
        print(f"\n[CONFIG-TUNING] Strategy: Reduce false positives (over-detection)")
        
        # Increase color ratio thresholds (requires more red/yellow to trigger)
        if 'fault_red_ratio' in updated_config:
            old_val = updated_config['fault_red_ratio']
            updated_config['fault_red_ratio'] = old_val * INCREASE_FACTOR
            adjustments_made.append(f"fault_red_ratio: {old_val:.6f} -> {updated_config['fault_red_ratio']:.6f}")
        
        if 'potential_yellow_ratio' in updated_config:
            old_val = updated_config['potential_yellow_ratio']
            updated_config['potential_yellow_ratio'] = old_val * INCREASE_FACTOR
            adjustments_made.append(f"potential_yellow_ratio: {old_val:.6f} -> {updated_config['potential_yellow_ratio']:.6f}")
        
        # Increase histogram distance threshold
        if 'hist_distance_min' in updated_config:
            old_val = updated_config['hist_distance_min']
            updated_config['hist_distance_min'] = old_val * INCREASE_FACTOR
            adjustments_made.append(f"hist_distance_min: {old_val:.6f} -> {updated_config['hist_distance_min']:.6f}")
        
        # Increase minimum area (filter out smaller detections)
        if 'min_blob_area_px' in updated_config:
            old_val = updated_config['min_blob_area_px']
            updated_config['min_blob_area_px'] = int(old_val * AREA_INCREASE)
            adjustments_made.append(f"min_blob_area_px: {old_val} -> {updated_config['min_blob_area_px']}")
        
        if 'min_cluster_area_px' in updated_config:
            old_val = updated_config['min_cluster_area_px']
            updated_config['min_cluster_area_px'] = int(old_val * AREA_INCREASE)
            adjustments_made.append(f"min_cluster_area_px: {old_val} -> {updated_config['min_cluster_area_px']}")
    
    # Case 2: Too many false negatives (missed detections)
    # Decrease thresholds to make detection more sensitive
    elif fn_count > 0 and fn_count > fp_count:
        print(f"\n[CONFIG-TUNING] Strategy: Reduce false negatives (missed detection)")
        
        # Decrease color ratio thresholds (requires less red/yellow to trigger)
        if 'fault_red_ratio' in updated_config:
            old_val = updated_config['fault_red_ratio']
            updated_config['fault_red_ratio'] = old_val * DECREASE_FACTOR
            adjustments_made.append(f"fault_red_ratio: {old_val:.6f} -> {updated_config['fault_red_ratio']:.6f}")
        
        if 'potential_yellow_ratio' in updated_config:
            old_val = updated_config['potential_yellow_ratio']
            updated_config['potential_yellow_ratio'] = old_val * DECREASE_FACTOR
            adjustments_made.append(f"potential_yellow_ratio: {old_val:.6f} -> {updated_config['potential_yellow_ratio']:.6f}")
        
        # Decrease histogram distance threshold
        if 'hist_distance_min' in updated_config:
            old_val = updated_config['hist_distance_min']
            updated_config['hist_distance_min'] = old_val * DECREASE_FACTOR
            adjustments_made.append(f"hist_distance_min: {old_val:.6f} -> {updated_config['hist_distance_min']:.6f}")
        
        # Decrease minimum area (allow smaller detections)
        if 'min_blob_area_px' in updated_config:
            old_val = updated_config['min_blob_area_px']
            updated_config['min_blob_area_px'] = max(10, int(old_val * AREA_DECREASE))
            adjustments_made.append(f"min_blob_area_px: {old_val} -> {updated_config['min_blob_area_px']}")
        
        if 'min_cluster_area_px' in updated_config:
            old_val = updated_config['min_cluster_area_px']
            updated_config['min_cluster_area_px'] = max(50, int(old_val * AREA_DECREASE))
            adjustments_made.append(f"min_cluster_area_px: {old_val} -> {updated_config['min_cluster_area_px']}")
    
    # Case 3: Balanced but not perfect - minor adjustments based on overall accuracy
    elif fp_count > 0 or fn_count > 0:
        print(f"\n[CONFIG-TUNING] Strategy: Balanced adjustment")
        net_adjustment = (fn_count - fp_count) / max(1, edited_fault_count)
        
        if abs(net_adjustment) > 0.1:  # Only adjust if >10% imbalance
            factor = DECREASE_FACTOR if net_adjustment > 0 else INCREASE_FACTOR
            small_factor = 1 + (factor - 1) * 0.5  # Half the adjustment
            
            if 'fault_red_ratio' in updated_config:
                old_val = updated_config['fault_red_ratio']
                updated_config['fault_red_ratio'] = old_val * small_factor
                adjustments_made.append(f"fault_red_ratio: {old_val:.6f} -> {updated_config['fault_red_ratio']:.6f}")
    
    else:
        print(f"\n[CONFIG-TUNING] Perfect match - no adjustments needed")
    
    # Print adjustments
    if adjustments_made:
        print(f"\n[CONFIG-TUNING] Adjustments applied:")
        for adj in adjustments_made:
            print(f"  - {adj}")
    else:
        print(f"\n[CONFIG-TUNING] No adjustments made")
    
    return updated_config, edited_fault_count, original_fault_count


@app.route("/update-config", methods=['POST'])
def api_update_config():
    """
    POST /update-config
    Receives configuration update request with:
    - baseline_image: baseline image file
    - maintenance_image: maintenance/current image file  
    - config: JSON file with current stored configuration
    - anomaly_results: JSON file with edited anomaly detection results (with manual changes)
    - original_anomaly_results: JSON file with original anomaly detection results from classifier
    
    Returns updated configuration parameters to be saved in database.
    
    Expected multipart/form-data:
        - baseline_image: baseline image file
        - maintenance_image: maintenance/current image file
        - config: JSON file with current stored configuration
        - anomaly_results: JSON file with edited fault regions and detection results
        - original_anomaly_results: JSON file with original fault regions from classifier
    """
    import random
    import time
    
    start_time = time.time()
    
    # Print received files for debugging
    print(f"[UPDATE-CONFIG] Received files: {list(request.files.keys())}")
    print(f"[UPDATE-CONFIG] Request form data: {list(request.form.keys())}")
    
    # Validate required files
    if "baseline_image" not in request.files:
        return jsonify({"error": "Missing 'baseline_image' file"}), 400
    if "maintenance_image" not in request.files:
        return jsonify({"error": "Missing 'maintenance_image' file"}), 400
    if "config" not in request.files:
        return jsonify({"error": "Missing 'config' JSON file"}), 400
    if "anomaly_results" not in request.files:
        return jsonify({"error": "Missing 'anomaly_results' JSON file"}), 400
    if "original_anomaly_results" not in request.files:
        return jsonify({"error": "Missing 'original_anomaly_results' JSON file"}), 400
    
    baseline_image = request.files["baseline_image"]
    maintenance_image = request.files["maintenance_image"]
    f_config = request.files["config"]
    f_anomaly_results = request.files["anomaly_results"]
    f_original_anomaly_results = request.files["original_anomaly_results"]
    
    # Print file details
    print(f"[UPDATE-CONFIG] baseline_image: {baseline_image.filename}, content_type: {baseline_image.content_type}")
    print(f"[UPDATE-CONFIG] maintenance_image: {maintenance_image.filename}, content_type: {maintenance_image.content_type}")
    print(f"[UPDATE-CONFIG] config: {f_config.filename}, content_type: {f_config.content_type}")
    print(f"[UPDATE-CONFIG] anomaly_results: {f_anomaly_results.filename}, content_type: {f_anomaly_results.content_type}")
    print(f"[UPDATE-CONFIG] original_anomaly_results: {f_original_anomaly_results.filename}, content_type: {f_original_anomaly_results.content_type}")
    
    # Temporary workspace for this request
    with tempfile.TemporaryDirectory() as td:
        # Save uploaded images
        baseline_ext = os.path.splitext(baseline_image.filename or "")[1] or ".png"
        maintenance_ext = os.path.splitext(maintenance_image.filename or "")[1] or ".png"
        
        baseline_path = os.path.join(td, f"baseline{baseline_ext}")
        maintenance_path = os.path.join(td, f"maintenance{maintenance_ext}")
        
        baseline_image.save(baseline_path)
        maintenance_image.save(maintenance_path)
        
        # Get file sizes for metrics
        baseline_size = baseline_image.content_length if hasattr(baseline_image, 'content_length') else os.path.getsize(baseline_path)
        maintenance_size = maintenance_image.content_length if hasattr(maintenance_image, 'content_length') else os.path.getsize(maintenance_path)
        
        # Parse JSON files
        try:
            stored_config = json.load(f_config.stream)
            anomaly_results = json.load(f_anomaly_results.stream)
            original_anomaly_results = json.load(f_original_anomaly_results.stream)
            
            # Print received configs and annotations
            print("\n" + "="*80)
            print("[UPDATE-CONFIG] RECEIVED CONFIG:")
            print(json.dumps(stored_config, indent=2))
            print("\n" + "="*80)
            print("[UPDATE-CONFIG] RECEIVED EDITED ANOMALY RESULTS:")
            print(json.dumps(anomaly_results, indent=2))
            print("\n" + "="*80)
            print("[UPDATE-CONFIG] RECEIVED ORIGINAL ANOMALY RESULTS:")
            print(json.dumps(original_anomaly_results, indent=2))
            print("="*80 + "\n")
            
            # Validate anomaly_results format
            if not isinstance(anomaly_results, dict):
                return jsonify({"error": "anomaly_results must be a dictionary"}), 400
            if not isinstance(original_anomaly_results, dict):
                return jsonify({"error": "original_anomaly_results must be a dictionary"}), 400
            
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Error parsing input files: {str(e)}"}), 400
        
        # Call the update function
        try:
            updated_config, edited_fault_count, original_fault_count = update_config_parameters(
                baseline_image_path=baseline_path,
                maintenance_image_path=maintenance_path,
                stored_config=stored_config,
                anomaly_results=anomaly_results,
                original_anomaly_results=original_anomaly_results
            )
            
            # Calculate training duration
            end_time = time.time()
            training_duration_ms = int((end_time - start_time) * 1000)
            
            # Calculate user corrections
            regions_deleted = original_fault_count - edited_fault_count
            
            # Print output config for debugging
            print("\n" + "="*80)
            print("[UPDATE-CONFIG] OUTPUT CONFIG:")
            print(json.dumps(updated_config, indent=2))
            print(f"[UPDATE-CONFIG] Original fault regions: {original_fault_count}")
            print(f"[UPDATE-CONFIG] Edited fault regions: {edited_fault_count}")
            print(f"[UPDATE-CONFIG] User corrections (deletions): {regions_deleted}")
            print("="*80 + "\n")
            
            # TODO: Save updated_config to database here
            # For now, just return it to the backend
            
            response = {
                "status": "success",
                "message": f"Model trained successfully. Analyzed {edited_fault_count} fault regions (originally {original_fault_count}) and optimized configuration parameters.",
                "updated_config": updated_config,
                "training_metrics": {
                    "fault_regions_analyzed": edited_fault_count,
                    "original_fault_regions": original_fault_count,
                    "user_corrections": regions_deleted,
                    "baseline_image_size": baseline_size,
                    "maintenance_image_size": maintenance_size,
                    "training_duration_ms": training_duration_ms
                }
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": f"Failed to update configuration: {str(e)}"
            }), 500


@app.route("/health", methods=['GET'])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    # Run the Flask app - Production mode (no debug for speed)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

