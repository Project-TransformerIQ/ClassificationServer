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

def update_config_parameters(baseline_image_path: str, 
                             maintenance_image_path: str, 
                             stored_config: dict, 
                             anomaly_results: dict) -> tuple:
    """
    Dummy function to update configuration parameters based on:
    - baseline_image_path: Path to the baseline image
    - maintenance_image_path: Path to the maintenance/current image
    - stored_config: Configuration parameters stored in the database
    - anomaly_results: Anomaly detection results with fault regions
    
    Returns:
        tuple: (stored_config dict, fault_count int)
    
    TODO: Implement actual logic to:
    1. Analyze baseline vs maintenance images
    2. Compare fault regions with current detection thresholds
    3. Adjust config parameters to improve detection accuracy
    4. Calculate optimal thresholds based on anomaly results data
    """
    # Extract fault regions count from anomaly results
    fault_regions = anomaly_results.get("fault_regions", [])
    fault_count = len(fault_regions)
    
    # Return the config as-is without any modifications
    return stored_config, fault_count


@app.route("/update-config", methods=['POST'])
def api_update_config():
    """
    POST /update-config
    Receives configuration update request with:
    - baseline_image: baseline image file
    - maintenance_image: maintenance/current image file  
    - config: JSON file with current stored configuration
    - anomaly_results: JSON file with anomaly detection results including fault regions
    
    Returns updated configuration parameters to be saved in database.
    
    Expected multipart/form-data:
        - baseline_image: baseline image file
        - maintenance_image: maintenance/current image file
        - config: JSON file with current stored configuration
        - anomaly_results: JSON file with fault regions and detection results
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
    
    baseline_image = request.files["baseline_image"]
    maintenance_image = request.files["maintenance_image"]
    f_config = request.files["config"]
    f_anomaly_results = request.files["anomaly_results"]
    
    # Print file details
    print(f"[UPDATE-CONFIG] baseline_image: {baseline_image.filename}, content_type: {baseline_image.content_type}")
    print(f"[UPDATE-CONFIG] maintenance_image: {maintenance_image.filename}, content_type: {maintenance_image.content_type}")
    print(f"[UPDATE-CONFIG] config: {f_config.filename}, content_type: {f_config.content_type}")
    print(f"[UPDATE-CONFIG] anomaly_results: {f_anomaly_results.filename}, content_type: {f_anomaly_results.content_type}")
    
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
            
            # Print received configs and annotations
            print("\n" + "="*80)
            print("[UPDATE-CONFIG] RECEIVED CONFIG:")
            print(json.dumps(stored_config, indent=2))
            print("\n" + "="*80)
            print("[UPDATE-CONFIG] RECEIVED ANOMALY RESULTS:")
            print(json.dumps(anomaly_results, indent=2))
            print("="*80 + "\n")
            
            # Validate anomaly_results format
            if not isinstance(anomaly_results, dict):
                return jsonify({"error": "anomaly_results must be a dictionary"}), 400
            
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Error parsing input files: {str(e)}"}), 400
        
        # Call the update function
        try:
            updated_config, fault_count = update_config_parameters(
                baseline_image_path=baseline_path,
                maintenance_image_path=maintenance_path,
                stored_config=stored_config,
                anomaly_results=anomaly_results
            )
            
            # Calculate training duration
            end_time = time.time()
            training_duration_ms = int((end_time - start_time) * 1000)
            
            # Print output config for debugging
            print("\n" + "="*80)
            print("[UPDATE-CONFIG] OUTPUT CONFIG:")
            print(json.dumps(updated_config, indent=2))
            print("="*80 + "\n")
            
            # TODO: Save updated_config to database here
            # For now, just return it to the backend
            
            response = {
                "status": "success",
                "message": f"Model trained successfully. Analyzed {fault_count} fault regions and optimized configuration parameters.",
                "updated_config": updated_config,
                "training_metrics": {
                    "fault_regions_analyzed": fault_count,
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

