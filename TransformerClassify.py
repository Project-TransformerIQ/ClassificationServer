from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import cv2, json, os

# ============================== CONFIG ==============================
CFG: Dict = {
    # --- ΔV (brightness) sensitivity ---
    "delta_k_sigma": 0.28,
    "delta_abs_min": 5,

    # --- morphology / cleanup ---
    "min_blob_area_px": 24,
    "open_iters": 1,
    "dilate_iters": 2,
    "keep_component_min_ratio": 8e-6,

    # --- decision thresholds ---
    "fault_red_ratio":        7e-5,
    "fault_red_min_pixels":   100,
    "potential_yellow_ratio": 1.2e-4,
    "fullwire_hot_fraction":  0.05,

    # --- wire-like heuristic ---
    "elongated_aspect_ratio": 3.0,

    # --- cluster nearby detections into one box ---
    "merge_close_frac": 0.02,
    "min_cluster_area_px": 100,

    # --- AUTO SIDEBAR (colorbar) detection ---
    "sidebar_search_frac": 0.25,
    "sidebar_min_width_frac": 0.02,
    "sidebar_max_width_frac": 0.18,
    "sidebar_min_valid_frac": 0.45,
    "sidebar_hue_span_deg": 80,
    "sidebar_margin_px": 2,

    # --- overlay masking ---
    "text_bottom_band_frac": 0.16,
    "mask_top_left_overlay": True,
    "top_left_box": (0.0, 0.0, 0.5, 0.12),

    # --- histogram / background deltas ---
    "h_bins": 36,
    "hist_distance_min": 0.06,
    "red_bg_ratio_min_increase": 0.15,
    "red_bg_min_abs": 0.001,
    "roi_s_min": 40, "roi_v_min": 35,

    # --- background (blue/black) ---
    "blue_h_lo": 90, "blue_h_hi": 140,
    "blue_s_min": 40, "blue_v_min": 30,
    "black_v_hi": 55,


    # === WHITE BACKGROUND (NEW) ===
    "white_bg_S_max": 35,       # background-ish white (pale; not a hot core)
    "white_bg_V_min": 245,
    "white_bg_exclude_near_warm_px": 9,  # exclude whites close to heat
    "white_bg_column_frac": 0.92,        # column must be ≥92% white to trim from side
    "white_bg_row_frac":    0.92,        # row must be ≥92% white to trim from top/bottom
    "crop_sidebar_in_output": True,      # still crop sidebars
    "crop_white_bg_in_output": True,     # crop large white background margins
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ============================= HELPERS ==============================
def mean_std_scalar(img: np.ndarray) -> Tuple[float, float]:
    mu_arr, sigma_arr = cv2.meanStdDev(img)
    return float(mu_arr.item()), float(sigma_arr.item())

def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists(): return []
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: p.name)

def pick_baseline(normal_imgs: List[Path]) -> Path:
    if not normal_imgs:
        raise RuntimeError("No images in normal/ to choose a baseline.")
    cands = [p for p in normal_imgs if "normal_001" in p.name.lower()]
    return sorted(cands)[0] if cands else normal_imgs[0]

# ====================== COLOR & BACKGROUND MASKS ======================
def hot_color_masks_relaxed(hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (red_orange, yellow, hot_all) in {0,255},
    including white cores that are embedded in/connected to hot regions.
    """

    # --- Red and orange ---
    red1 = cv2.inRange(hsv, (0,   100, 110), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 100, 110), (180,255, 255))
    red  = cv2.bitwise_or(red1, red2)

    orange = cv2.inRange(hsv, (10,  90, 110), (20, 255, 255))
    red_orange = cv2.bitwise_or(red, orange)

    # --- Yellow ---
    yellow = cv2.inRange(hsv, (20,  60, 110), (38, 255, 255))

    # --- White-hot candidate (very bright, low saturation) ---
    white_hot = cv2.inRange(hsv, (0, 0, 230), (180, 50, 255))

    # --- Halo expansion: grow red/orange so it overlaps with white cores ---
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    red_orange_halo = cv2.dilate(red_orange, k)

    # --- Only keep white cores connected to red/orange halo ---
    warm_plus_white = cv2.bitwise_or(red_orange_halo, white_hot)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(warm_plus_white, 8)

    white_core = np.zeros_like(white_hot)
    for i in range(1, num):  # skip background
        comp = (lab == i)
        if np.any(red_orange[comp] > 0):  # valid only if connected to red/orange
            white_core[comp] = white_hot[comp]

    # --- Final hot mask ---
    hot_all = cv2.bitwise_or(cv2.bitwise_or(red_orange, yellow), white_core)

    return red_orange, yellow, hot_all



def background_masks(hsv: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    blue = cv2.inRange(
        hsv,
        (cfg["blue_h_lo"], cfg["blue_s_min"], cfg["blue_v_min"]),
        (cfg["blue_h_hi"], 255,                255)
    )
    v = hsv[:,:,2]
    black = np.zeros_like(v, np.uint8); black[v <= cfg["black_v_hi"]] = 255
    return blue, black

# ====================== TEXT / OVERLAY MASKING ======================
def text_overlay_mask(bgr: np.ndarray, cfg: Dict) -> np.ndarray:
    H, W = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # warm_txt  = cv2.inRange(hsv, (5,  150, 150), (25, 255,255))
    # cyan_txt  = cv2.inRange(hsv, (80,  80, 100), (100,255,255))
    # green_txt = cv2.inRange(hsv, (35,  60,  80), (85, 255,255))
    # color_mask = warm_txt | cyan_txt | green_txt

    mser_mask = np.zeros((H, W), np.uint8)
    mser = cv2.MSER_create()
    regions, boxes = mser.detectRegions(gray)
    mask_mser = np.zeros_like(gray)
    for (x, y, w, h) in boxes:
        if w > 3 and h > 8:  # filter tiny
            cv2.rectangle(mask_mser, (x, y), (x + w, y + h), 255, -1)
    # mser = cv2.MSER_create(20, 30, 5000)
    # regions, boxes = mser.detectRegions(gray)
    # for (x,y,w,h) in boxes:
    #     if w <= 2 or h <= 7: continue
    #     ar = w / max(1, h)
    #     if ar < 0.15 or ar > 12.0: continue
    #     if h > int(0.2 * H): continue
    #     cv2.rectangle(mser_mask, (x,y), (x+w,y+h), 255, -1)

    bottom_band = np.zeros((H, W), np.uint8)
    h0 = int(H * (1.0 - cfg["text_bottom_band_frac"]))
    bottom_band[h0:, :] = 255

    tl_mask = np.zeros((H, W), np.uint8)
    if cfg.get("mask_top_left_overlay", False):
        x0f, y0f, wf, hf = cfg["top_left_box"]
        x0, y0 = int(W*x0f), int(H*y0f)
        w_box, h_box = int(W*wf), int(H*hf)
        cv2.rectangle(tl_mask, (x0,y0), (x0+w_box,y0+h_box), 255, -1)


    right_mask = np.zeros((H, W), np.uint8)

    # fraction of width to cut (15%)
    cut_frac = 0.20

    # compute box
    w_box = int(W * cut_frac)  # 15% of width
    h_box = H  # full height
    x0 = W - w_box  # start at right edge
    y0 = 0

    # draw mask
    cv2.rectangle(right_mask, (x0, y0), (W, y0 + h_box), 255, -1)
    # text_mask = color_mask | mser_mask | bottom_band | tl_mask
    # text_mask =  bottom_band | tl_mask
    text_mask =  mser_mask | bottom_band |right_mask
    k = np.ones((3,3), np.uint8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, k, iterations=1)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN,  k, iterations=1)
    return text_mask

# =================== AUTO SIDEBAR (COLORBAR) DETECTION ===================
def detect_sidebar_mask(hsv: np.ndarray, txt_mask: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Optional[slice]]:
    H, W = hsv.shape[:2]
    S, V, Hh = hsv[:,:,1], hsv[:,:,2], hsv[:,:,0]
    icol = int(W * cfg["sidebar_search_frac"])
    x_left_zone  = range(0, max(1, icol))
    x_right_zone = range(W - icol, W)

    colorful = ((S >= cfg["roi_s_min"]) & (V >= cfg["roi_v_min"]) & (txt_mask == 0)).astype(np.uint8)

    def column_is_legend(x):
        col = colorful[:, x].astype(bool)
        if col.mean() < cfg["sidebar_min_valid_frac"]:
            return False
        hvals = Hh[col]
        if hvals.size < max(32, int(0.2*H)):
            return False
        p5  = float(np.percentile(hvals, 5))
        p95 = float(np.percentile(hvals, 95))
        return (p95 - p5) >= cfg["sidebar_hue_span_deg"]

    def pick_run(zone):
        flags = [1 if column_is_legend(x) else 0 for x in zone]
        runs, start = [], None
        for i, f in enumerate(flags):
            if f and start is None: start = i
            if (not f or i == len(flags)-1) and start is not None:
                end = i if not f else i
                runs.append((start, end)); start = None
        if not runs: return None
        runs = sorted(runs, key=lambda r: (r[1]-r[0]+1), reverse=True)
        r0 = runs[0]
        x0 = zone.start + r0[0]
        x1 = zone.start + r0[1]
        return (x0, x1)

    left  = pick_run(x_left_zone)
    right = pick_run(x_right_zone)
    def score(r): return -1 if r is None else (r[1]-r[0]+1)
    pick = left if score(left) >= score(right) else right

    mask = np.zeros((H, W), np.uint8)
    crop = None
    if pick is not None:
        x0 = max(0, pick[0] - cfg["sidebar_margin_px"])
        x1 = min(W, pick[1] + 1 + cfg["sidebar_margin_px"])
        minw = int(cfg["sidebar_min_width_frac"] * W)
        maxw = int(cfg["sidebar_max_width_frac"] * W)
        if (x1 - x0) >= minw and (x1 - x0) <= maxw:
            mask[:, x0:x1] = 255
            crop = (slice(None), slice(x0, x1))
    return mask, crop

# ==================== ROI & HISTOGRAM COMPARISON ====================
def roi_mask_from_image(hsv: np.ndarray, txt_mask: np.ndarray, cfg: Dict) -> np.ndarray:
    S, V = hsv[:,:,1], hsv[:,:,2]
    roi = np.zeros_like(V, np.uint8)
    roi[(S >= cfg["roi_s_min"]) & (V >= cfg["roi_v_min"])] = 255
    roi[txt_mask > 0] = 0
    return roi

def hue_hist(hsv: np.ndarray, mask: np.ndarray, bins: int) -> np.ndarray:
    H = hsv[:,:,0]
    hist = cv2.calcHist([H], [0], mask, [bins], [0,180])
    hist = hist / (hist.sum() + 1e-9)
    return hist

def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    bc = np.sum(np.sqrt(h1 * h2))
    return float(np.clip(np.sqrt(1.0 - bc), 0.0, 1.0))

# ====================== BACKGROUND CONTRAST CHECK ======================
def contrastful_new_red_mask(new_red: np.ndarray, hsv_curr: np.ndarray, cfg: Dict) -> np.ndarray:
    blue, black = background_masks(hsv_curr, cfg)
    bg = cv2.bitwise_or(blue, black)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    bg_local = cv2.filter2D((bg > 0).astype(np.float32), -1, k/(k.sum()))
    out = np.zeros_like(new_red)
    out[(new_red > 0) & (bg_local >= 0.5)] = 255
    return out

# ===================== MERGE NEARBY DETECTIONS =====================
def merge_close_components(mask: np.ndarray, cfg: Dict) -> np.ndarray:
    H, W = mask.shape[:2]
    radius = max(1, int(min(H, W) * cfg["merge_close_frac"]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    num, lab, st, _ = cv2.connectedComponentsWithStats(merged, 8)
    clean = np.zeros_like(mask)
    for i in range(1, num):
        if st[i, cv2.CC_STAT_AREA] >= cfg["min_cluster_area_px"]:
            clean[lab == i] = 255
    return clean


# ============== WHITE BACKGROUND (NEW): MASK & CROPPING EDGES ==============
def white_background_mask(hsv: np.ndarray, warm_mask: np.ndarray, txt_mask: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    Detect big WHITE background (very bright, low saturation) that touches image edges
    and is NOT near warm colors nor text overlays.
    """
    H, W = hsv.shape[:2]
    S, V = hsv[:,:,1], hsv[:,:,2]
    white = ((S <= cfg["white_bg_S_max"]) & (V >= cfg["white_bg_V_min"])).astype(np.uint8) * 255

    # Exclude warm neighborhood (so white-hot cores are kept for faults)
    if cfg["white_bg_exclude_near_warm_px"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*cfg["white_bg_exclude_near_warm_px"]+1,)*2)
        warm_neigh = cv2.dilate(warm_mask, k)
        white[warm_neigh > 0] = 0

    # Exclude text overlays
    white[txt_mask > 0] = 0

    # Keep only components that touch any border (true margins)
    num, lab, st, _ = cv2.connectedComponentsWithStats(white, 8)
    mask = np.zeros_like(white)
    for i in range(1, num):
        x,y,w,h = st[i,0], st[i,1], st[i,2], st[i,3]
        if x == 0 or y == 0 or (x+w) >= W or (y+h) >= H:
            mask[lab == i] = 255
    return mask

def compute_white_bg_crop_edges(white_bg_mask: np.ndarray, cfg: Dict) -> Tuple[int,int,int,int]:
    """
    Return cropping edges (left, right, top, bottom) in pixels to trim only white background.
    Uses per-edge coverage thresholds so black scene is preserved.
    """
    H, W = white_bg_mask.shape[:2]
    col_cov = white_bg_mask.mean(axis=0) / 255.0  # coverage per column
    row_cov = white_bg_mask.mean(axis=1) / 255.0  # coverage per row

    # from left
    xL = 0
    for x in range(W):
        if col_cov[x] >= cfg["white_bg_column_frac"]:
            xL = x + 1
        else:
            break
    # from right
    xR = W
    for x in range(W-1, -1, -1):
        if col_cov[x] >= cfg["white_bg_column_frac"]:
            xR = x
        else:
            break
    # from top
    yT = 0
    for y in range(H):
        if row_cov[y] >= cfg["white_bg_row_frac"]:
            yT = y + 1
        else:
            break
    # from bottom
    yB = H
    for y in range(H-1, -1, -1):
        if row_cov[y] >= cfg["white_bg_row_frac"]:
            yB = y
        else:
            break
    # sanity
    if xL >= xR: xL, xR = 0, W
    if yT >= yB: yT, yB = 0, H
    return xL, xR, yT, yB

def remove_text_inpaint(
    bgr: np.ndarray,
    save_clean_path: Optional[str] = None,
    inpaint_radius: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove ONLY COLORED text overlays (do NOT remove white/gray text).
    Pipeline:
      1) MSER -> text-shaped region proposals (geometry filtered)
      2) Keep only COLORED pixels inside those regions (high saturation)
      3) Inpaint just those colored text pixels

    Returns (clean_image, colored_text_mask). Optionally saves the cleaned image.
    """
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    area_img = float(H * W)
    k3 = np.ones((3, 3), np.uint8)

    # Precompute edges for a simple "textiness" measure
    edges = cv2.Canny(gray, 60, 140)

    # Helper gates
    def box_ok(w: int, h: int) -> bool:
        if w < 6 or h < 10: return False
        if h > int(0.30 * H): return False
        ar = w / float(h)
        return 0.18 <= ar <= 12.0

    def stroke_width_median(bin_comp: np.ndarray) -> float:
        # bin_comp is 0/255 mask
        dt = cv2.distanceTransform(bin_comp, cv2.DIST_L2, 3)
        vals = dt[bin_comp > 0]
        return 2.0 * float(np.median(vals)) if vals.size else 1e9

    def edge_density(x0, y0, w, h) -> float:
        e = edges[y0:y0+h, x0:x0+w]
        return float(np.count_nonzero(e)) / max(1.0, float(w * h))

    # 1) Text proposals via MSER (color-agnostic)
    #    We will filter to colored pixels later.
    mser = cv2.MSER_create(8, 30, 6000)  # (delta, min_area, max_area) — tuned for small UI text
    regions, boxes = mser.detectRegions(gray)

    text_region_mask = np.zeros((H, W), np.uint8)

    for pts in regions:
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(pts)
        if not box_ok(w, h):
            continue

        # Fill the region (use hull to be robust)
        hull = cv2.convexHull(pts)
        comp = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(comp, hull, 255)

        # Geometry guards to keep things "text-like"
        A = float(np.count_nonzero(comp))
        if A < 40:
            continue
        if A / area_img > 0.03:  # reject huge regions
            continue

        sw = stroke_width_median(comp)
        ed = edge_density(x, y, w, h)

        # Text tends to have thin–medium strokes and decent edge density
        if sw <= 6.0 and ed >= 0.045:
            text_region_mask = cv2.bitwise_or(text_region_mask, comp)

    # Optional tidy
    text_region_mask = cv2.morphologyEx(text_region_mask, cv2.MORPH_CLOSE, k3, iterations=1)

    # 2) Keep ONLY colored pixels inside those text regions
    #    Define "colored" as sufficiently saturated & bright (not white/gray/black).
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    colored_pixels = ((S >= 70) & (V >= 70)).astype(np.uint8) * 255  # broad: any hue, but truly colored

    # (Optional) If you want to exclude blues/cyans etc., restrict hue ranges here instead.

    colored_text_mask = cv2.bitwise_and(text_region_mask, colored_pixels)

    # Tighten to the actual colored strokes (reduce halos)
    colored_text_mask = cv2.morphologyEx(colored_text_mask, cv2.MORPH_OPEN, k3, iterations=1)

    # Remove tiny specks (avoid over-inpainting)
    num, lab, st, _ = cv2.connectedComponentsWithStats(colored_text_mask, 8)
    clean_mask = np.zeros_like(colored_text_mask)
    for i in range(1, num):
        area = st[i, cv2.CC_STAT_AREA]
        if area >= 15:  # keep small but non-trivial
            clean_mask[lab == i] = 255

    # 3) Inpaint JUST those colored text pixels
    clean = cv2.inpaint(bgr, clean_mask, inpaint_radius, cv2.INPAINT_TELEA)

    if save_clean_path is not None:
        Path(save_clean_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_clean_path, clean)

    return clean, clean_mask
def make_side_margin_mask(shape_or_img, frac: float = 0.15) -> np.ndarray:
    """
    Return a binary mask (255 = IGNORE) covering the left & right 'frac' of width.
    shape_or_img: (H, W) tuple or an image array with shape (H, W, [C])
    """
    if isinstance(shape_or_img, tuple):
        H, W = shape_or_img
    else:
        H, W = shape_or_img.shape[:2]

    w = int(round(W * frac))
    m = np.zeros((H, W), np.uint8)
    if w > 0:
        m[:, :w] = 255           # left 15%
        m[:, W - w:] = 255       # right 15%
    return m



# ===================== PAIRWISE CLASSIFICATION =====================


def classify_transformer(
    baseline_path: str,
    current_path: str,
    out_path: str,
    cfg: Dict = CFG
) -> Dict:
    # --- Load images ---
    base = cv2.imread(baseline_path, cv2.IMREAD_COLOR)
    curr = cv2.imread(current_path, cv2.IMREAD_COLOR)
    if base is None or curr is None:
        raise FileNotFoundError(f"Could not read one of:\n{baseline_path}\n{current_path}")

    # --- Resize current to match baseline dimensions ---
    H, W = base.shape[:2]
    curr = cv2.resize(curr, (W, H))

    # --- Keep original resized image for drawing later ---
    curr_raw = curr.copy()
    side_ignore = make_side_margin_mask(curr, frac=0.15)
    # Directory to save intermediate processed (inpainted) images
    inpaint_dir = Path(out_path).parent / "inpainted"
    inpaint_dir.mkdir(parents=True, exist_ok=True)

    curr_clean_path = str(inpaint_dir / "current_inpainted.png")
    base_clean_path = str(inpaint_dir / "baseline_inpainted.png")

    curr, txt_c = remove_text_inpaint(curr.copy(), save_clean_path=curr_clean_path)
    base, txt_b = remove_text_inpaint(base, save_clean_path=base_clean_path)


    # Convert to HSV
    hsv_b = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
    v_b, v_c = hsv_b[:,:,2], hsv_c[:,:,2]

    # Sidebar mask
    sidebar_mask_c, _ = detect_sidebar_mask(hsv_c, txt_c, CFG)

    # Thermal color masks
    red_orange_b, yellow_b, _ = hot_color_masks_relaxed(hsv_b)
    red_orange_c, yellow_c, _ = hot_color_masks_relaxed(hsv_c)
    warm_c = cv2.bitwise_or(red_orange_c, yellow_c)

    # White background mask
    white_bg_c = white_background_mask(hsv_c, warm_c, txt_c, CFG)

    # Remove unwanted regions
    remove_mask_c = cv2.bitwise_or(cv2.bitwise_or(sidebar_mask_c, white_bg_c), side_ignore)

    for m in (red_orange_b, yellow_b, red_orange_c, yellow_c, v_b, v_c, txt_b, txt_c):
        m[remove_mask_c > 0] = 0
        m[txt_b > 0] = 0
        m[txt_c > 0] = 0

    warm_c = cv2.bitwise_or(red_orange_c, yellow_c)

    # --- ΔV gate (brightness difference baseline vs current) ---
    v1 = cv2.GaussianBlur(v_b, (5,5), 0)
    v2 = cv2.GaussianBlur(v_c, (5,5), 0)
    dv = cv2.subtract(v2, v1)
    mu, sigma = mean_std_scalar(dv)
    thr = max(float(CFG["delta_abs_min"]), mu + CFG["delta_k_sigma"] * sigma)
    _, dmask = cv2.threshold(dv, thr, 255, cv2.THRESH_BINARY)
    dmask[(txt_c > 0) | (remove_mask_c > 0)] = 0

    # --- Color gain (new red/yellow) ---
    red_gain = cv2.bitwise_and(red_orange_c, cv2.bitwise_not(red_orange_b))
    yel_gain = cv2.bitwise_and(yellow_c,    cv2.bitwise_not(yellow_b))

    new_red = cv2.bitwise_or(cv2.bitwise_and(red_orange_c, dmask), red_gain)
    new_yel = cv2.bitwise_or(cv2.bitwise_and(yellow_c,    dmask), yel_gain)
    new_hot = cv2.bitwise_or(new_red, new_yel)

    # --- Morphological cleanup ---
    k = np.ones((3,3), np.uint8)
    for m in (new_red, new_yel, new_hot):
        cv2.morphologyEx(m, cv2.MORPH_OPEN, k, dst=m, iterations=CFG["open_iters"])
        cv2.dilate(m, k, dst=m, iterations=CFG["dilate_iters"])

    def keep_reasonable(mask: np.ndarray) -> np.ndarray:
        num, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        area_img = mask.shape[0]*mask.shape[1]
        for i in range(1, num):
            area = st[i, cv2.CC_STAT_AREA]
            if area >= CFG["min_blob_area_px"] and area/area_img >= CFG["keep_component_min_ratio"]:
                out[lab == i] = 255
        return out

    new_red  = keep_reasonable(new_red)
    new_yel  = keep_reasonable(new_yel)
    new_hot  = keep_reasonable(new_hot)


    # --- Histogram comparison ---
    roi_b = roi_mask_from_image(hsv_b, txt_b, CFG); roi_b[remove_mask_c > 0] = 0
    roi_c = roi_mask_from_image(hsv_c, txt_c, CFG); roi_c[remove_mask_c > 0] = 0
    h_b = hue_hist(hsv_b, roi_b, CFG["h_bins"])
    h_c = hue_hist(hsv_c, roi_c, CFG["h_bins"])
    hist_dist = bhattacharyya(h_b, h_c)

    # --- Background ratios ---
    blue_b, black_b = background_masks(hsv_b, CFG)
    blue_c, black_c = background_masks(hsv_c, CFG)
    bg_b = cv2.bitwise_or(blue_b, black_b); bg_b[(txt_b > 0) | (remove_mask_c > 0)] = 0
    bg_c = cv2.bitwise_or(blue_c, black_c); bg_c[(txt_c > 0) | (remove_mask_c > 0)] = 0

    eps = 1e-9
    red_b_ratio_to_bg = float(cv2.countNonZero(red_orange_b)) / (float(cv2.countNonZero(bg_b)) + eps)
    red_c_ratio_to_bg = float(cv2.countNonZero(red_orange_c)) / (float(cv2.countNonZero(bg_c)) + eps)
    red_bg_ratio_increase = red_c_ratio_to_bg - red_b_ratio_to_bg

    # --- Contrastful red ---
    contrast_red = contrastful_new_red_mask(new_red, hsv_c, CFG)
    contrast_red_pixels = int(cv2.countNonZero(contrast_red))

    # --- Area ratios ---
    Hc, Wc = curr.shape[:2]
    area = float(Hc*Wc)
    red_pixels    = int(cv2.countNonZero(new_red))
    yellow_pixels = int(cv2.countNonZero(new_yel))
    hot_pixels    = int(cv2.countNonZero(new_hot))
    red_ratio     = red_pixels    / area
    yellow_ratio  = yellow_pixels / area
    hot_ratio     = hot_pixels    / area

    # --- Elongated blobs (wire-like) ---
    def elongated(mask: np.ndarray) -> bool:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (w2, h2) = cv2.minAreaRect(c)[1]
            ar = max(w2, h2) / max(1e-3, min(w2, h2))
            if ar >= cfg["elongated_aspect_ratio"]:
                return True
        return False
    red_elon = elongated(new_red)
    yel_elon = elongated(new_yel)

    # --- Decision rules ---
    faulty_rule = (
        (red_ratio >= cfg["fault_red_ratio"]) or
        (red_pixels >= cfg["fault_red_min_pixels"]) or
        (contrast_red_pixels >= cfg["fault_red_min_pixels"] // 2) or
        (hist_dist >= cfg["hist_distance_min"] and
         red_bg_ratio_increase >= cfg["red_bg_ratio_min_increase"] * max(red_b_ratio_to_bg, 1.0) and
         red_c_ratio_to_bg >= cfg["red_bg_min_abs"]) or
        red_elon
    )
    potentially_rule = (
        (yellow_ratio >= cfg["potential_yellow_ratio"]) or
        yel_elon or
        (hot_ratio >= cfg["fullwire_hot_fraction"] and red_ratio < yellow_ratio)
    )

    if faulty_rule:
        label = "Faulty"
        subtype = "Overload (wire-like)" if red_elon else "Loose joint / hotspot"
        chosen_mask = new_red
        box_color = (0,0,255); tag = "FAULT"
    elif potentially_rule:
        label = "Potentially Faulty"
        subtype = "Full-wire warm-up" if (hot_ratio >= cfg["fullwire_hot_fraction"] and red_ratio < yellow_ratio) else "Point/patch warm-up"
        chosen_mask = new_yel
        box_color = (0,255,255); tag = "POTENTIAL"
    else:
        label = "Normal"; subtype = "No significant change"
        chosen_mask = new_yel
        box_color = (0,255,255); tag = ""

    # --- Merge close detections ---
    merged_mask = merge_close_components(chosen_mask, cfg)

    # --- Draw results on ORIGINAL resized image ---
    # --- Draw results on ORIGINAL resized image ---
    vis = curr_raw.copy()
    Hc, Wc = vis.shape[:2]
    cnts, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        # pad each box a bit
        pad = max(cfg.get("box_min_pad_px", 4),
                  int(round(cfg.get("box_pad_frac", 0.08) * max(w, h))))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(Wc, x + w + pad)
        y1 = min(Hc, y + h + pad)

        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))

        if tag:
            cv2.rectangle(
                vis, (int(x0), int(y0)), (int(x1), int(y1)),
                box_color, int(cfg.get("box_thickness", 3))
            )
            cv2.putText(
                vis, tag, (int(x0), max(0, int(y0) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA
            )
    # --- Header text ---
    header = f"Classification: {label} ({subtype})"
    cv2.putText(vis, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    info1 = f"Δhot: {hot_ratio:.2%} | red: {red_ratio:.2%} | yellow: {yellow_ratio:.2%}"
    cv2.putText(vis, info1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    info2 = f"Hdist: {hist_dist:.3f} | red/bg B→C: {red_b_ratio_to_bg:.4f}->{red_c_ratio_to_bg:.4f} (+{red_bg_ratio_increase:.4f})"
    cv2.putText(vis, info2, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # --- Save result (no cropping) ---
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)

    return {
        "classification": label,
        "subtype_guess": subtype,
        "boxes": boxes,
        "area_hot": hot_ratio,
        "area_red": red_ratio,
        "area_yellow": yellow_ratio,
        "hist_dist": hist_dist,
        "red_bg_baseline": red_b_ratio_to_bg,
        "red_bg_current":  red_c_ratio_to_bg,
        "output_image": out_path
    }


def classify_transformer_api(
    baseline_path: str,
    current_path: str,
    out_path: str,
    cfg: Dict = CFG
) -> Dict:
    # --- Load images (same inputs as before) ---
    base = cv2.imread(baseline_path, cv2.IMREAD_COLOR)
    curr = cv2.imread(current_path,  cv2.IMREAD_COLOR)
    if base is None or curr is None:
        raise FileNotFoundError(f"Could not read one of:\n{baseline_path}\n{current_path}")

    # --- Resize current to match baseline dimensions ---
    H, W = base.shape[:2]
    curr = cv2.resize(curr, (W, H))

    # --- Keep original resized image ONLY for dimensions (no drawing/saving) ---
    Hc, Wc = curr.shape[:2]
    side_ignore = make_side_margin_mask(curr, frac=0.15)

    # --- Inpaint colored text overlays (NO file saves) ---
    curr, txt_c = remove_text_inpaint(curr.copy(), save_clean_path=None)
    base, txt_b = remove_text_inpaint(base,        save_clean_path=None)

    # --- HSV conversion ---
    hsv_b = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
    v_b, v_c = hsv_b[:,:,2], hsv_c[:,:,2]

    # --- Sidebar mask ---
    sidebar_mask_c, _ = detect_sidebar_mask(hsv_c, txt_c, cfg)

    # --- Thermal color masks ---
    red_orange_b, yellow_b, _ = hot_color_masks_relaxed(hsv_b)
    red_orange_c, yellow_c, _ = hot_color_masks_relaxed(hsv_c)
    warm_c = cv2.bitwise_or(red_orange_c, yellow_c)

    # --- White background mask ---
    white_bg_c = white_background_mask(hsv_c, warm_c, txt_c, cfg)

    # --- Remove unwanted regions (sidebar, white bg, side margins, text) ---
    remove_mask_c = cv2.bitwise_or(cv2.bitwise_or(sidebar_mask_c, white_bg_c), side_ignore)
    for m in (red_orange_b, yellow_b, red_orange_c, yellow_c, v_b, v_c, txt_b, txt_c):
        m[remove_mask_c > 0] = 0
        m[txt_b > 0] = 0
        m[txt_c > 0] = 0

    warm_c = cv2.bitwise_or(red_orange_c, yellow_c)

    # --- ΔV gate (brightness difference baseline vs current) ---
    v1 = cv2.GaussianBlur(v_b, (5,5), 0)
    v2 = cv2.GaussianBlur(v_c, (5,5), 0)
    dv = cv2.subtract(v2, v1)
    mu, sigma = mean_std_scalar(dv)
    thr = max(float(cfg["delta_abs_min"]), mu + cfg["delta_k_sigma"] * sigma)
    _, dmask = cv2.threshold(dv, thr, 255, cv2.THRESH_BINARY)
    dmask[(txt_c > 0) | (remove_mask_c > 0)] = 0

    # --- Color gain (new red/yellow) ---
    red_gain = cv2.bitwise_and(red_orange_c, cv2.bitwise_not(red_orange_b))
    yel_gain = cv2.bitwise_and(yellow_c,    cv2.bitwise_not(yellow_b))
    new_red  = cv2.bitwise_or(cv2.bitwise_and(red_orange_c, dmask), red_gain)
    new_yel  = cv2.bitwise_or(cv2.bitwise_and(yellow_c,    dmask), yel_gain)
    new_hot  = cv2.bitwise_or(new_red, new_yel)

    # --- Morphological cleanup ---
    k = np.ones((3,3), np.uint8)
    for m in (new_red, new_yel, new_hot):
        cv2.morphologyEx(m, cv2.MORPH_OPEN, k, dst=m, iterations=cfg["open_iters"])
        cv2.dilate(m, k, dst=m, iterations=cfg["dilate_iters"])

    def keep_reasonable(mask: np.ndarray) -> np.ndarray:
        num, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        area_img = mask.shape[0]*mask.shape[1]
        for i in range(1, num):
            area = st[i, cv2.CC_STAT_AREA]
            if area >= cfg["min_blob_area_px"] and area/area_img >= cfg["keep_component_min_ratio"]:
                out[lab == i] = 255
        return out

    new_red  = keep_reasonable(new_red)
    new_yel  = keep_reasonable(new_yel)
    new_hot  = keep_reasonable(new_hot)

    # --- Histogram comparison ---
    roi_b = roi_mask_from_image(hsv_b, txt_b, cfg); roi_b[remove_mask_c > 0] = 0
    roi_c = roi_mask_from_image(hsv_c, txt_c, cfg); roi_c[remove_mask_c > 0] = 0
    h_b = hue_hist(hsv_b, roi_b, cfg["h_bins"])
    h_c = hue_hist(hsv_c, roi_c, cfg["h_bins"])
    hist_dist = bhattacharyya(h_b, h_c)

    # --- Background ratios ---
    blue_b, black_b = background_masks(hsv_b, cfg)
    blue_c, black_c = background_masks(hsv_c, cfg)
    bg_b = cv2.bitwise_or(blue_b, black_b); bg_b[(txt_b > 0) | (remove_mask_c > 0)] = 0
    bg_c = cv2.bitwise_or(blue_c, black_c); bg_c[(txt_c > 0) | (remove_mask_c > 0)] = 0

    eps = 1e-9
    red_b_ratio_to_bg = float(cv2.countNonZero(red_orange_b)) / (float(cv2.countNonZero(bg_b)) + eps)
    red_c_ratio_to_bg = float(cv2.countNonZero(red_orange_c)) / (float(cv2.countNonZero(bg_c)) + eps)
    red_bg_ratio_increase = red_c_ratio_to_bg - red_b_ratio_to_bg  # computed but not returned (schema unchanged)

    # --- Contrastful red (used in rules only) ---
    contrast_red = contrastful_new_red_mask(new_red, hsv_c, cfg)
    contrast_red_pixels = int(cv2.countNonZero(contrast_red))

    # --- Area ratios ---
    area = float(Hc*Wc)
    red_pixels    = int(cv2.countNonZero(new_red))
    yellow_pixels = int(cv2.countNonZero(new_yel))
    hot_pixels    = int(cv2.countNonZero(new_hot))
    red_ratio     = red_pixels    / area
    yellow_ratio  = yellow_pixels / area
    hot_ratio     = hot_pixels    / area

    # --- Elongated blobs (wire-like) ---
    def elongated(mask: np.ndarray) -> bool:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (w2, h2) = cv2.minAreaRect(c)[1]
            ar = max(w2, h2) / max(1e-3, min(w2, h2))
            if ar >= cfg["elongated_aspect_ratio"]:
                return True
        return False
    red_elon = elongated(new_red)
    yel_elon = elongated(new_yel)

    # --- Decision rules ---
    faulty_rule = (
        (red_ratio >= cfg["fault_red_ratio"]) or
        (red_pixels >= cfg["fault_red_min_pixels"]) or
        (contrast_red_pixels >= cfg["fault_red_min_pixels"] // 2) or
        (hist_dist >= cfg["hist_distance_min"] and
         red_bg_ratio_increase >= cfg["red_bg_ratio_min_increase"] * max(red_b_ratio_to_bg, 1.0) and
         red_c_ratio_to_bg >= cfg["red_bg_min_abs"]) or
        red_elon
    )
    potentially_rule = (
        (yellow_ratio >= cfg["potential_yellow_ratio"]) or
        yel_elon or
        (hot_ratio >= cfg["fullwire_hot_fraction"] and red_ratio < yellow_ratio)
    )

    if faulty_rule:
        label = "Faulty"
        subtype = "Overload (wire-like)" if red_elon else "Loose joint / hotspot"
        chosen_mask = new_red
    elif potentially_rule:
        label = "Potentially Faulty"
        subtype = "Full-wire warm-up" if (hot_ratio >= cfg["fullwire_hot_fraction"] and red_ratio < yellow_ratio) \
                  else "Point/patch warm-up"
        chosen_mask = new_yel
    else:
        label = "Normal"
        subtype = "No significant change"
        chosen_mask = new_yel  # same as before

    # --- Merge close detections & produce boxes (NO drawing; keep boxes for all labels) ---
    merged_mask = merge_close_components(chosen_mask, cfg)
    cnts, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        pad = max(cfg.get("box_min_pad_px", 4),
                  int(round(cfg.get("box_pad_frac", 0.08) * max(w, h))))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(Wc, x + w + pad)
        y1 = min(Hc, y + h + pad)
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))

    # --- IMPORTANT: NO saving and NO drawing ---
    # (We still return out_path for schema compatibility.)
    return {
        "classification": label,
        "subtype_guess": subtype,
        "boxes": boxes,                 # list of (x,y,w,h) tuples, same as before
        "area_hot": hot_ratio,
        "area_red": red_ratio,
        "area_yellow": yellow_ratio,
        "hist_dist": hist_dist,
        "red_bg_baseline": red_b_ratio_to_bg,
        "red_bg_current":  red_c_ratio_to_bg,
        "output_image": out_path        # unchanged key/value shape, but no file is written
    }


# ========================== BATCH RUNNER ==========================
def run_dataset(root: str, out_root: str, cfg: Dict = CFG) -> pd.DataFrame:
    """
    Writes, for each transformer and each candidate image, a folder containing:
      - a copy of the picked baseline image (baseline.<ext>)
      - a copy of the candidate image (candidate.<ext>)
      - the annotated result (candidate_annotated.png)

    Also writes out_root/summary.csv with the same columns you already use.
    """
    from pathlib import Path
    from typing import Dict, Tuple, List
    import pandas as pd
    import json, shutil

    root_p = Path(root); out_p = Path(out_root)
    if out_p.exists():
        shutil.rmtree(out_p)
    out_p.mkdir(parents=True, exist_ok=True)

    rows = []
    # iterate transformers
    for tx_dir in sorted([p for p in root_p.iterdir() if p.is_dir()]):
        normal_dir, faulty_dir = tx_dir / "normal", tx_dir / "faulty"
        normal_imgs, faulty_imgs = list_images(normal_dir), list_images(faulty_dir)

        if not normal_imgs:
            print(f"[WARN] No normal images in {tx_dir}. Skipping.")
            continue

        # choose baseline from normals
        baseline = pick_baseline(normal_imgs)
        baseline_ext = baseline.suffix.lower()

        # build (label, candidate_path) pairs
        pairs: List[Tuple[str, Path]] = []
        for img in normal_imgs:
            if img == baseline:
                continue
            pairs.append(("normal", img))
        for img in faulty_imgs:
            pairs.append(("faulty", img))

        # process each comparison
        for sub, img in pairs:
            pair_dir = out_p / tx_dir.name / img.stem
            pair_dir.mkdir(parents=True, exist_ok=True)

            # copy originals into the same folder
            baseline_copy = pair_dir / f"baseline{baseline_ext}"
            candidate_copy = pair_dir / f"candidate{img.suffix.lower()}"
            if not baseline_copy.exists():
                shutil.copy2(baseline, baseline_copy)
            shutil.copy2(img, candidate_copy)

            # annotated output path (in the same folder)
            out_img = pair_dir / "candidate_annotated.png"

            # run classification using the local copies
            res = classify_transformer(str(baseline_copy), str(candidate_copy), str(out_img), cfg)

            # record row (keep your original columns; update annotated_image path)
            rows.append({
                "transformer": tx_dir.name,
                "baseline": str(baseline.relative_to(root_p)),
                "image":     str(img.relative_to(root_p)),
                "pred_label": res["classification"],
                "subtype":    res["subtype_guess"],
                "area_hot":   res["area_hot"],
                "area_red":   res["area_red"],
                "area_yellow":res["area_yellow"],
                "hist_dist":  res["hist_dist"],
                "red_bg_baseline": res["red_bg_baseline"],
                "red_bg_current":  res["red_bg_current"],
                "boxes":      json.dumps(res["boxes"]),
                # path relative to out_root
                "annotated_image": str(out_img.relative_to(out_p)),
            })

    df = pd.DataFrame(rows, columns=[
        "transformer","baseline","image","pred_label","subtype",
        "area_hot","area_red","area_yellow",
        "hist_dist","red_bg_baseline","red_bg_current",
        "boxes","annotated_image"
    ])
    csv_path = out_p / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[DONE] Wrote: {csv_path}")
    return df




# ============================ USAGE ============================
# Example:
DATASET_ROOT = r"D:\Software\SoftwareDesign\Transformer Images Reduced"     # T1..T13 each with normal/ & faulty/
OUTPUT_ROOT  = r"D:\Software\SoftwareDesign\OutPut Images"
# df = run_dataset(DATASET_ROOT, OUTPUT_ROOT, CFG)
# print(df.head())
