# Transformer Anomaly Detection Server

A high-performance Flask server that provides AI-powered thermal imaging analysis for transformer fault detection. The server uses computer vision techniques to detect hotspots, warm patches, and other anomalies in thermal images.

## 🔍 Detection Approach Overview

### TransformerClassify Algorithm

Our detection system employs a sophisticated multi-stage analysis approach:

1. **Thermal Image Preprocessing**

   - Text removal and inpainting to clean thermal images
   - Sidebar detection and masking for measurement legends
   - Color space conversion (BGR → HSV) for thermal analysis

2. **Comparative Analysis**

   - Baseline vs. current image comparison
   - Brightness delta (ΔV) sensitivity analysis
   - Histogram distance computation for change detection

3. **Fault Classification**

   - **FAULT (Hotspots)**: High-temperature anomalies (red regions)

     - Red area ratio analysis
     - Background temperature increase detection
     - Confidence based on thermal metrics

   - **POTENTIAL (Warm Patches)**: Medium-temperature concerns (yellow regions)
     - Yellow area ratio evaluation
     - Overall heat distribution analysis
     - Wire connectivity assessment

4. **Morphological Processing**

   - Blob detection and cleanup
   - Component analysis with minimum area thresholds
   - Bounding box generation and region characterization

5. **Confidence Scoring**
   - Multi-factor confidence calculation
   - Thermal signature strength assessment
   - Historical baseline comparison

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

### System Requirements

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 500MB for dependencies

## 🛠️ Configuration

### Performance Settings

- **Debug Mode**: Disabled for production speed
- **Threading**: Enabled for concurrent requests
- **Port**: 5000 (configurable in app.py)
- **Host**: 0.0.0.0 (accepts external connections)

### Algorithm Parameters (CFG)

Key detection thresholds can be adjusted in `TransformerClassify.py`:

- `fault_red_ratio`: Minimum red area ratio for fault detection
- `potential_yellow_ratio`: Minimum yellow area ratio for potential faults
- `delta_k_sigma`: Brightness change sensitivity
- `min_blob_area_px`: Minimum blob size for detection

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

## 🚨 Troubleshooting

### Common Issues

1. **Server won't start**

   - Check if port 5000 is available
   - Ensure virtual environment is activated
   - Verify all dependencies are installed

2. **Import errors**

   - Check Python path and virtual environment
   - Install missing packages: `pip install -r requirements.txt`

3. **Memory issues**

   - Large thermal images may require more RAM
   - Consider image resizing for performance

4. **400 Bad Request**
   - Ensure both baseline and candidate images are provided
   - Check file format (JPG, PNG supported)
   - Verify multipart/form-data content type

## 📈 Performance Optimization

- **Production Mode**: Debug disabled for faster processing
- **Threaded Requests**: Multiple concurrent image analyses
- **Temporary Files**: Automatic cleanup after processing
- **Memory Management**: Efficient image handling with OpenCV

## 🔒 Security Notes

- Server accepts external connections (0.0.0.0)
- No authentication implemented (add as needed)
- Temporary files are cleaned up automatically
- Consider adding rate limiting for production use

**🚀 Ready to detect transformer anomalies with AI-powered thermal analysis!**

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

## Example Response

The `/fault-regions` endpoint returns thermal imaging fault data including:

- Fault region details (hotspots, warm patches)
- Bounding box coordinates
- Color information
- Confidence scores
- Display metadata

## Usage

```bash
# Get fault regions data
curl http://localhost:5000/fault-regions

# Get server info
curl http://localhost:5000/
```
