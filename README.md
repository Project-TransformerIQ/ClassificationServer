# Transformer Anomaly Detection Server

A high-performance Flask server that provides AI-powered thermal imaging analysis for transformer fault detection. The server uses computer vision techniques to detect hotspots, warm patches, and other anomalies in thermal images.

## 🔍 Detection Approach Overview

# Algorithm: Thermal Fault Detection (Baseline vs Current)

## Overview

Given a **baseline** thermal image *B* and a **current** image *C*, the system removes colored UI overlays, masks non‑fault screen areas (side legends & white margins), detects **new heat** (color + brightness rise), cleans/merges detections, applies rule‑based decisions, draws boxes, and returns results (image + JSON) via a Flask API.

---

## Inputs & Outputs

**Inputs**

* Baseline image **B**
* Current image **C**
* Config **CFG** (thresholds, morphology, masks)

**Classifier Outputs** (Python dict)

* `classification ∈ {"Normal", "Potentially Faulty", "Faulty"}`
* `subtype_guess`
* `boxes = [(x, y, w, h), …]` — on **C**
* Metrics: `area_red`, `area_yellow`, `area_hot`, `hist_dist`, `red_bg_baseline`, `red_bg_current`
* `output_image` — path to annotated **C** with drawn boxes

---

## Step‑by‑step Algorithm

1. **Align**

   * Resize **C** to **B**’s dimensions; keep `C_raw` for final drawing.

2. **Erase colored overlays (inpaint)**

   * Run **MSER** on grayscale to propose text‑like components.
   * Filter by geometry & structure: min size, max height (≤ 0.3·H), aspect ratio range, edge density, stroke‑width median.
   * Inside those components, keep **colored pixels only** (`S ≥ 70` and `V ≥ 70`).
   * **Inpaint** only those colored pixels (Telea) → `B_clean`, `C_clean` and text masks `txt_b`, `txt_c`.

3. **Mask non‑fault regions**

   * **Sidebar/legend** detection (left/right zones): columns are “legend” if colorful, dense, and show large **hue span** → `sidebar_mask`.
   * **White margins**: bright, low‑saturation components that touch image edges and are **not near warm colors** and **not text** → `white_bg_mask`.
   * **Side margins**: ignore left & right **15%** of width → `side_ignore`.
   * Combined: `remove_mask = sidebar_mask ∪ white_bg_mask ∪ side_ignore`.

4. **Thermal colors (HSV)**

   * From `B_clean` and `C_clean`, extract:

     * `red_orange` (red bands + orange),
     * `yellow`, and
     * **white‑hot cores** that are connected to a dilated warm halo.
   * Zero out any pixels belonging to `txt_*` and `remove_mask`.

5. **Brightness increase (ΔV gate)**

   * Blur V channels; compute `dv = V_C − V_B`.
   * Adaptive threshold: `thr = max(delta_abs_min, mean(dv) + delta_k_sigma · std(dv))`.
   * `dmask = (dv > thr)` minus text/removed areas.

6. **New heat candidates**

   * `red_gain = red_orange_C ∧ ¬red_orange_B`
   * `yel_gain = yellow_C     ∧ ¬yellow_B`
   * `new_red = (red_orange_C ∧ dmask) ∪ red_gain`
   * `new_yel = (yellow_C     ∧ dmask) ∪ yel_gain`
   * `new_hot = new_red ∪ new_yel`

7. **Cleanup & merging**

   * Morphology: **open + dilate** on `new_red`, `new_yel`, `new_hot`.
   * Drop tiny/insignificant components (absolute area and image‑ratio guards).
   * **Merge** close blobs (radius = `merge_close_frac * min(H, W)`); keep clusters ≥ `min_cluster_area_px`.

8. **Scene comparison & background**

   * **ROI** (colorful & bright, not masked) → hue histograms for B and C.
   * **Histogram distance**: `hist_dist = Bhattacharyya(h_B, h_C)`.
   * **Background ratios** (blue ∪ black = background):

     * `red_bg_baseline = |red_orange_B| / |bg_B|`
     * `red_bg_current  = |red_orange_C| / |bg_C|`
     * `red_bg_increase = red_bg_current − red_bg_baseline`
   * **Contrastful red**: retain `new_red` pixels that sit on local background to favor genuine hot spots.

9. **Areas & shape**

   * `area_red    = |new_red| / (H · W)`
   * `area_yellow = |new_yel| / (H · W)`
   * `area_hot    = |new_hot| / (H · W)`
   * **Elongation**: min‑area‑rect aspect ≥ `elongated_aspect_ratio` → wire‑like.

10. **Decision rules**

* **Faulty** if **any**:

  * `area_red ≥ fault_red_ratio`, **or** red pixels ≥ `fault_red_min_pixels`;
  * **or** contrastful‑red pixels large (≥ ~½·`fault_red_min_pixels`);
  * **or** (`hist_dist ≥ hist_distance_min` **and**
    `red_bg_increase ≥ red_bg_ratio_min_increase · max(red_bg_baseline, 1)` **and**
    `red_bg_current ≥ red_bg_min_abs`);
  * **or** **red elongated** (wire‑like).
* **Potentially Faulty** if **any**:

  * `area_yellow ≥ potential_yellow_ratio`,
  * **or** **yellow elongated**,
  * **or** full‑wire warm‑up: `area_hot ≥ fullwire_hot_fraction` and `area_red < area_yellow`.
* Else → **Normal**.
* Choose mask for boxes: `new_red` (Faulty) or `new_yel` (Potential).

11. **Boxes & annotation**

* Find contours on merged mask; **pad** each box (via `box_pad_frac` or `box_min_pad_px`).
* Draw rectangles + tag; render header metrics; save annotated image.

12. **Return**

* `{ classification, subtype_guess, boxes, area_hot, area_red, area_yellow, hist_dist, red_bg_baseline, red_bg_current, output_image }`.

---

## API (Flask)

**POST** `/detect-anomalies`

* Form fields: `baseline` (file), `candidate` (file)
* Optional query: `?returnAnnotated=1` to include annotated path in JSON

**Response JSON**

* `fault_regions`: list of regions built from `boxes`, each with:

  * `type` ("Hotspot" for FAULT, "Warmspot" for POTENTIAL)
  * `dominantColor`, `colorRgb`
  * `boundingBox {x, y, width, height, areaPx}`
  * `centroid {x, y}`
  * `aspectRatio`, `elongated`, `connectedToWire` (proxy = elongated)
  * `tag ∈ {"FAULT","POTENTIAL","NORMAL"}`
  * `confidence` (heuristic blend of metrics by tag)
* `display_metadata`: `{ boxColors, timestamp }`
* `annotated_image` (present only if `?returnAnnotated=1`)

**Example**

```bash
curl -X POST "http://localhost:5000/detect-anomalies?returnAnnotated=1" \
  -F "baseline=@/path/to/baseline.png" \
  -F "candidate=@/path/to/current.png"
```

---

## Complexity

~O(H · W) per image pair (MSER + morphology + histograms); suitable for single‑image request latency.

---

## Key Tuning Knobs (CFG)

* **Sensitivity**: `delta_k_sigma`, `delta_abs_min`
* **Strictness**: `fault_red_ratio`, `fault_red_min_pixels`, `potential_yellow_ratio`
* **Clutter control**: `sidebar_*`, `white_bg_*`, side margin fraction (0.15)
* **Box look & feel**: `box_pad_frac`, `box_min_pad_px`, `box_thickness`


## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone/Navigate to the project directory:**

   ```powershell
   cd "c:\Users\niros\Desktop\Software Design Project\ClassificationServer"
   ```

2. **Create and activate virtual environment:**

   ```powershell
   # Create virtual environment
   python -m venv .venv

   # Activate (Windows PowerShell)
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Server

### Standard Run Command

```powershell
cd "c:\Users\niros\Desktop\Software Design Project\ClassificationServer"
& "C:/Users/niros/Desktop/Software Design Project/ClassificationServer/.venv/Scripts/python.exe" app.py
```

### Alternative Methods

```powershell
# Method 1: Using virtual environment directly
.\.venv\Scripts\python.exe app.py

# Method 2: Activate environment first
.\.venv\Scripts\Activate.ps1
python app.py
```

The server will start on:

- **Primary URL**: `http://localhost:5000`
- **Network Access**: `http://0.0.0.0:5000` (accepts external connections)

## 📡 API Endpoints

### 1. Anomaly Detection (Main Endpoint)

- **URL**: `POST /detect-anomalies`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `baseline`: Baseline thermal image file
  - `candidate`: Current thermal image file for comparison
  - `returnAnnotated` (optional): Include annotated image path in response

### 2. Health Check

- **URL**: `GET /health`
- **Response**: Server status confirmation

## 📊 Response Format

```json
{
  "fault_regions": [
    {
      "dbId": 1,
      "regionId": 1,
      "type": "Hotspot",
      "dominantColor": "red",
      "colorRgb": [255, 0, 0],
      "boundingBox": {
        "x": 412,
        "y": 226,
        "width": 38,
        "height": 40,
        "areaPx": 1520
      },
      "centroid": { "x": 431, "y": 246 },
      "aspectRatio": 0.95,
      "elongated": false,
      "connectedToWire": false,
      "tag": "FAULT",
      "confidence": 0.95
    }
  ],
  "display_metadata": {
    "id": 1,
    "boxColors": {
      "FAULT": "255,0,0",
      "POTENTIAL": "255,255,0",
      "NORMAL": "0,255,0"
    },
    "timestamp": "2025-10-05T13:15:00"
  }
}
```

## 📦 Dependencies

### Core Dependencies

```
Flask==3.0.0           # Web framework
numpy                  # Numerical computing
pandas                 # Data manipulation
opencv-python          # Computer vision
requests==2.31.0       # HTTP client (for testing)
```

## 🧪 Testing

### Test Health Endpoint

```powershell
Invoke-WebRequest -Uri "http://localhost:5000/health" -Method GET
```

### Test Anomaly Detection

```bash
curl -X POST http://localhost:5000/detect-anomalies \
  -F "baseline=@path/to/baseline_image.jpg" \
  -F "candidate=@path/to/current_image.jpg"
```

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## Endpoints

- `GET /` - Home page with basic info
- `GET /fault-regions` - Returns fault regions data in JSON format

## Usage

```bash
# Get fault regions data
curl http://localhost:5000/fault-regions

# Get server info
curl http://localhost:5000/
```
