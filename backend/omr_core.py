"""
OMR Core Processing Module

This module handles the core optical mark recognition (OMR) functionality.
It processes scanned answer sheets and detects student responses by analyzing
filled bubble patterns.

Key Features:
- Image preprocessing and deskewing
- Circle detection for answer bubbles
- Mark analysis and answer prediction
- Visual overlay generation for verification

Author: Giri Krishna
Last Updated: September 2025
"""

import cv2
import numpy as np
import pandas as pd
import re
from io import BytesIO
import logging
from typing import Tuple, List, Dict, Any

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration Constants ====================

# Circle detection parameters (fine-tuned for typical OMR sheets)
CIRCLE_DETECTION_PARAMS = {
    'dp': 1.2,                    # Inverse ratio of accumulator resolution
    'min_distance': 18,           # Minimum distance between circle centers
    'param1': 60,                 # Upper threshold for edge detection
    'param2': 18,                 # Accumulator threshold for center detection
    'min_radius': 7,              # Minimum circle radius
    'max_radius': 16,             # Maximum circle radius
}

# Row grouping and analysis parameters
ROW_TOLERANCE = 10                # Y-coordinate tolerance for grouping circles into rows
INNER_CIRCLE_RATIO = 0.62        # Ratio for inner circle analysis
BACKGROUND_INNER_THRESHOLD = 0.85 # Inner area background threshold
BACKGROUND_OUTER_THRESHOLD = 1.25 # Outer area background threshold
MIN_FILL_THRESHOLD = 0.42         # Minimum fill ratio to consider a bubble marked
MIN_DELTA_THRESHOLD = 0.15        # Minimum difference for mark detection
TIE_MARGIN = 0.12                 # Margin for handling tied responses

# Test structure configuration
SUBJECTS = ["Python", "Data Analysis", "MySQL", "Power BI", "Adv Stats"]
NUM_SUBJECTS = 5
QUESTIONS_PER_SUBJECT = 20
ANSWER_CHOICES = ["A", "B", "C", "D"]

# ==================== Image Processing Functions ====================

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load and preprocess an image from raw bytes.
    
    Args:
        image_bytes: Raw image data in bytes format
        
    Returns:
        Preprocessed OpenCV image (BGR format)
        
    Raises:
        ValueError: If image data is invalid or corrupted
    """
    try:
        # Convert bytes to numpy array and decode image
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Unable to decode image - file may be corrupted or in unsupported format")
        
        # Normalize image size for consistent processing
        height, width = image.shape[:2]
        target_size = 1600
        scale_factor = target_size / max(height, width)
        
        if scale_factor < 1.0:
            # Resize large images to improve processing speed
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {str(e)}")
        raise ValueError(f"Image processing failed: {str(e)}")

def correct_image_skew(grayscale_image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect and correct skew in a grayscale image using line detection.
    
    Many scanned documents have slight rotation that can affect circle detection.
    This function uses Hough line detection to find predominant angles and
    corrects the image orientation.
    
    Args:
        grayscale_image: Input grayscale image
        
    Returns:
        Tuple of (corrected_image, rotation_angle_degrees)
    """
    try:
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        
        # Create binary image using adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
        
        # Detect edges for line detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None:
            logger.info("No lines detected for skew correction")
            return grayscale_image, 0.0
        
        # Analyze line angles to determine predominant skew
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_degrees = theta * 180 / np.pi
            # Focus on horizontal-ish lines (avoid vertical lines)
            if 45 < angle_degrees < 135:
                angles.append(angle_degrees - 90)
        
        if not angles:
            logger.info("No suitable lines found for skew correction")
            return grayscale_image, 0.0
        
        # Use median angle to avoid outliers
        correction_angle = float(np.median(angles))
        logger.info(f"Detected skew angle: {correction_angle:.2f} degrees")
        
        # Apply rotation correction
        height, width = grayscale_image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width//2, height//2), correction_angle, 1.0)
        corrected_image = cv2.warpAffine(
            grayscale_image, rotation_matrix, (width, height), 
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        
        return corrected_image, correction_angle
        
    except Exception as e:
        logger.error(f"Skew correction failed: {str(e)}")
        return grayscale_image, 0.0

def _binary_ink(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    return cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV,31,10)

def _detect_circles(gray):
    g = cv2.medianBlur(gray,3)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, DP, MIN_DIST,
                               param1=PARAM1, param2=PARAM2,
                               minRadius=MIN_R, maxRadius=MAX_R)
    return [] if circles is None else np.round(circles[0]).astype(int).tolist()

def _group_rows(circles):
    if not circles: return []
    circles.sort(key=lambda c:c[1])
    rows, cur = [], [circles[0]]
    for c in circles[1:]:
        if abs(c[1]-np.mean([p[1] for p in cur])) <= ROW_Y_TOL: cur.append(c)
        else: rows.append(cur); cur=[c]
    rows.append(cur)
    return [sorted(r, key=lambda c:c[0]) for r in rows if len(r)>=4]

def _ad_groups(row):
    if len(row) < 4: return []
    rs = [c[2] for c in row]; r_med = float(np.median(rs))
    gap_break = 3.2*r_med
    clusters, cur = [], [row[0]]
    for c in row[1:]:
        if (c[0]-cur[-1][0]) > gap_break: clusters.append(cur); cur=[c]
        else: cur.append(c)
    clusters.append(cur)
    groups=[]
    for cl in clusters:
        if len(cl) < 4: continue
        best, best_w = None, 1e9
        for i in range(0,len(cl)-3):
            chunk = sorted(cl[i:i+4], key=lambda c:c[0])
            xs=[p[0] for p in chunk]; ys=[p[1] for p in chunk]; rs4=[p[2] for p in chunk]
            width = max(xs)-min(xs)
            d1,d2,d3 = xs[1]-xs[0], xs[2]-xs[1], xs[3]-xs[2]
            d_med = np.median([d1,d2,d3])
            if np.std(ys) > 0.5*r_med: continue
            if not all(0.6*d_med <= d <= 1.4*d_med for d in [d1,d2,d3]): continue
            if (np.max(rs4)-np.min(rs4)) > 0.6*r_med: continue
            if not (2.6*r_med <= width <= 10*r_med): continue
            if width < best_w: best_w, best = width, chunk
        if best is not None: groups.append(best)
    groups.sort(key=lambda g:int(np.mean([p[0] for p in g])))
    return groups

def _kmeans_1d(xs, k=5, iters=30):
    xs = np.asarray(xs, dtype=np.float32).reshape(-1,1)
    if len(xs) < k:
        centers = np.linspace(xs.min(), xs.max(), k).reshape(-1,1)
    else:
        rng = np.random.RandomState(0)
        centers = xs[rng.choice(len(xs), k, replace=False)]
    for _ in range(iters):
        d = np.abs(xs - centers.T)
        labels = np.argmin(d, axis=1)
        new = []
        for c in range(k):
            pts = xs[labels==c]
            new.append(np.mean(pts) if len(pts) else centers[c])
        new = np.array(new).reshape(-1,1)
        if np.allclose(new, centers): break
        centers = new
    return labels, centers.reshape(-1)

def _cluster_subjects(groups):
    if not groups: return [[] for _ in range(NUM_SUBJECTS)]
    xs = [int(np.mean([p[0] for p in g])) for g in groups]
    labels, centers = _kmeans_1d(xs, k=NUM_SUBJECTS, iters=40)
    order = np.argsort(centers)
    label2idx = {lbl:i for i,lbl in enumerate(order)}
    buckets=[[] for _ in range(NUM_SUBJECTS)]
    for g,lbl in zip(groups,labels):
        cx = int(np.mean([p[0] for p in g])); cy = int(np.mean([p[1] for p in g]))
        buckets[label2idx[lbl]].append({"cx":cx,"cy":cy,"row":sorted(g,key=lambda c:c[0])})
    for b in buckets: b.sort(key=lambda G:G["cy"])
    return buckets

def _estimate_pitch(ys):
    ys = sorted(ys)
    if len(ys) < 5: return None
    diffs = np.diff(ys); diffs = [d for d in diffs if d>4]
    if not diffs: return None
    return float(np.median(diffs))

def _grid_mapping(ys, dy):
    ys = sorted(ys)
    if not ys: return (lambda y: None), None
    y_min = ys[0]
    best_y0, best_score, best_err = None, -1, 1e9
    for frac in np.linspace(-0.8, 0.0, 25):
        y0 = y_min + frac*dy
        bins, errs = [], []
        for y in ys:
            i = int(round((y - y0)/dy))
            if 0 <= i < Q_PER_SUBJECT:
                bins.append(i); errs.append(abs((y0 + i*dy) - y))
        score = len(set(bins)); err = np.mean(errs) if errs else 1e9
        if (score > best_score) or (score == best_score and err < best_err):
            best_score, best_err, best_y0 = score, err, y0
    def f(y):
        i = int(round((y - best_y0)/dy))
        return i if 0 <= i < Q_PER_SUBJECT else None
    return f, best_y0

def _ink_scores(ink, x,y,r):
    inner_r = max(2, int(r*INNER_RATIO))
    m_in = np.zeros_like(ink, dtype=np.uint8); cv2.circle(m_in, (x,y), inner_r, 255, -1)
    m_bg = np.zeros_like(ink, dtype=np.uint8)
    cv2.circle(m_bg,(x,y), int(r*BG_OUTER),255,-1); cv2.circle(m_bg,(x,y), int(r*BG_INNER),0,-1)
    iw = cv2.countNonZero(cv2.bitwise_and(ink, ink, mask=m_in))
    it = max(1, cv2.countNonZero(m_in))
    bw = cv2.countNonZero(cv2.bitwise_and(ink, ink, mask=m_bg))
    bt = max(1, cv2.countNonZero(m_bg))
    inner = iw/it
    delta = inner - (bw/bt)
    return inner, delta

def _pick_letter(ink, group):
    scores=[]
    for circ in group:
        inner, delta = _ink_scores(ink, circ[0],circ[1],circ[2])
        scores.append(inner if delta >= MIN_DELTA else 0.0)
    best = int(np.argmax(scores))
    top2 = sorted(scores, reverse=True)[:2] + [0.0]
    if top2[0] < MIN_FILL: return ""
    if (top2[0]-top2[1]) < MARGIN_FOR_TIES: return ""
    return CHOICES[best]

def _parse_key_df(df: pd.DataFrame):
    df.columns = [str(c).strip() for c in df.columns]
    def find_col(cands):
        for want in cands:
            for c in df.columns:
                if re.sub(r"\s+","",want.lower()) in re.sub(r"\s+","",c.lower()):
                    return c
        return None
    mapping = {
        "Python": ["Python"],
        "Data Analysis": ["EDA","Data Analysis"],
        "MySQL": ["SQL","MySQL"],
        "Power BI": ["POWER BI","Power BI"],
        "Adv Stats": ["Statistics","Satistics","Adv Stats"]
    }
    key={}
    for subj, cands in mapping.items():
        col = find_col(cands)
        answers=[]
        if col is None:
            answers=[set() for _ in range(Q_PER_SUBJECT)]
        else:
            for i in range(Q_PER_SUBJECT):
                if i >= len(df): answers.append(set()); continue
                cell = str(df.iloc[i][col]).lower()
                opts = set(x.upper() for x in re.findall(r"[abcd]", cell))
                answers.append(opts)
        key[subj]=answers
    return key

def grade_from_bytes(image_bytes: bytes, key_bytes: bytes):
    """Return (summary_dict, details_list, overlay_image_ndarray)."""
    # -- image --
    img = _imread_bytes(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray, _ = _deskew_gray(gray)
    ink = _binary_ink(gray)

    # -- key --
    try:
        df = pd.read_excel(BytesIO(key_bytes), engine="openpyxl")
    except Exception:
        try:
            df = pd.read_csv(BytesIO(key_bytes))
        except Exception:
            df = pd.read_csv(BytesIO(key_bytes), sep=";")
    key = _parse_key_df(df)

    # -- detect --
    circles = _detect_circles(gray)
    rows = _group_rows(circles)
    groups = []
    for r in rows: groups.extend(_ad_groups(r))
    subjects = _cluster_subjects(groups)

    # -- overlay as ndarray (always) --
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    pred_by_subject = {}
    details=[]
    for s_idx, subj_groups in enumerate(subjects):
        subj_name = SUBJECTS[s_idx]
        ys = [G["cy"] for G in subj_groups]
        dy = _estimate_pitch(ys)
        preds = [""]*Q_PER_SUBJECT
        if dy is not None and ys:
            to_idx, y0 = _grid_mapping(ys, dy)
            for G in subj_groups:
                i = to_idx(G["cy"])
                if i is None or i>=Q_PER_SUBJECT: continue
                letter = _pick_letter(ink, G["row"])
                preds[i]=letter
                for j,c in enumerate(G["row"]):
                    filled = (CHOICES[j]==letter)
                    color = (0,255,0) if filled else (0,0,255)
                    cv2.circle(overlay,(c[0],c[1]),c[2],color,2)
                    if filled:
                        cv2.circle(overlay,(c[0],c[1]),max(2,int(c[2]*0.45)),(0,255,0),-1)
        pred_by_subject[subj_name]=preds
        for q in range(1,Q_PER_SUBJECT+1):
            details.append({"Subject":subj_name, "QNo":q, "Pred":preds[q-1]})

    per_subj = {s: sum(1 for p,allowed in zip(pred_by_subject[s], key[s]) if p and p in allowed)
                for s in SUBJECTS}
    total = sum(per_subj.values())
    summary = {"total_correct": total}
    summary.update({f"{s}_score": per_subj[s] for s in SUBJECTS})

    # IMPORTANT: return overlay as NumPy array (never a string/path/base64)
    return summary, details, overlay
