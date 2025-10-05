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
    multipart/form-data with two files:
        - baseline:   baseline image file
        - candidate:  current image file
    Optional query/string:
        - returnAnnotated (0/1): include path string to annotated image in payload (debug)
    """
    if "baseline" not in request.files or "candidate" not in request.files:
        error_msg = f"Upload two images with keys 'baseline' and 'candidate'. Received: {list(request.files.keys())}"
        return jsonify({"error": error_msg}), 400

    f_base = request.files["baseline"]
    f_curr = request.files["candidate"]

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

        # Run your existing classifier (it writes annotated image to out_path)
        res = classify_transformer(b_path, c_path, out_path, CFG)

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

@app.route("/health", methods=['GET'])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    # Run the Flask app - Production mode (no debug for speed)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
