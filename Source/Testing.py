import os
import json
import cv2
import numpy as np
import torch
from main import build_model, TARGET_SIZE, USE_LINES # Assumes main.py is in the same folder

# -------- CONFIG --------
# Ensure this path points to the NEW model you trained with a smaller batch size
MODEL_PATH = r"C:\Users\Siddharth Kumar\Desktop\Hackathon\runway.pth" if USE_LINES else r"C:\Users\Siddharth Kumar\Desktop\Hackathon\runway_no_lines.pth"
IMAGE_PATH = r"C:\Users\Siddharth Kumar\Desktop\Hackathon\Hackathon SDG\HackSpace---Runway-model\DataSets\1920x1080\1920x1080\test\4AK606_1_6LDImage2.png"
OUT_DIR = "."
THRESH = 0.1
MORPH_KERNEL_SIZE = (7, 7)
ROW_SAMPLE_STEP = 4
# ------------------------

# --- Setup Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = build_model(use_lines=USE_LINES)
try:
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please check the path.")
    exit()
except Exception as e:
    print(f"Error loading model state: {e}")
    exit()


# --- Helper Functions ---
def clean_mask(mask_uint8, kernel_size):
    """Apply morphological operations to clean up a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Fill small holes
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels <= 1:
        return opened  # No components or only background
    
    # Find the index of the largest component (ignoring background at index 0)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_idx).astype(np.uint8) * 255
    return largest_mask


def compute_centerline(bin_mask, step):
    """Compute a smoothed centerline by finding the midpoint of each row."""
    h, w = bin_mask.shape
    centerline_pts = []
    for row in range(0, h, step):
        # Find the column indices where the mask is non-zero
        cols = np.where(bin_mask[row] > 0)[0]
        if cols.size > 0:
            left_col = cols[0]
            right_col = cols[-1]
            center_x = (left_col + right_col) // 2
            centerline_pts.append([center_x, row])

    if not centerline_pts:
        return []

    # Smooth the centerline using a moving average
    centerline_pts = np.array(centerline_pts, dtype=np.float32)
    window_size = max(5, len(centerline_pts) // 10) # Dynamic window size
    if window_size % 2 == 0: window_size += 1 # Must be odd

    if len(centerline_pts) > window_size:
        smoothed_x = cv2.GaussianBlur(centerline_pts[:, 0], (window_size, 1), 0).flatten()
        centerline_pts[:, 0] = smoothed_x

    return centerline_pts.astype(np.int32).tolist()


# --- Main Inference Pipeline ---
def predict():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    orig_h, orig_w = img_bgr.shape[:2]
    model_h, model_w = TARGET_SIZE # (1080, 1920)

    # 1. PREPROCESS IMAGE (MATCH TRAINING)
    # Resize the image to the exact size the model was trained on
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (model_w, model_h), interpolation=cv2.INTER_AREA)
    
    # Normalize and convert to tensor (C, H, W)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    input_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device, dtype=torch.float32)

    # 2. RUN MODEL
    with torch.no_grad():
        output = model(input_tensor) # (1, C, model_h, model_w)
        probs = torch.sigmoid(output).cpu().numpy().squeeze(0) # (C, model_h, model_w)

    # 3. EXTRACT AND PROCESS MASK
    # We only need the first channel (the area mask) for contours
    mask_prob = probs[0] if probs.ndim == 3 else probs
    
    # Threshold into a binary mask
    mask_bin = (mask_prob > THRESH).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUT_DIR, "pred_mask_resized_model_res.png"), mask_bin)

    # Clean up the mask
    mask_clean = clean_mask(mask_bin, MORPH_KERNEL_SIZE)
    cv2.imwrite(os.path.join(OUT_DIR, "pred_mask_clean.png"), mask_clean)
    
    # 4. FIND CONTOURS AND CENTERLINE (at model resolution)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No runway detected.")
        return

    runway_contour = max(contours, key=cv2.contourArea)
    
    # Compute centerline from the cleaned mask
    centerline_model_coords = compute_centerline((mask_clean > 0).astype(np.uint8), ROW_SAMPLE_STEP)

    # 5. SCALE COORDINATES BACK TO ORIGINAL IMAGE SIZE
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h

    # Scale the main contour and separate left/right points
    scaled_runway_pts = (runway_contour.squeeze(1) * [scale_x, scale_y]).astype(np.int32)
    
    # Find bounding box on scaled points to separate left and right
    x, y, w, h = cv2.boundingRect(scaled_runway_pts)
    cx = x + w // 2
    
    left_pts = sorted([p.tolist() for p in scaled_runway_pts if p[0] < cx], key=lambda pt: pt[1])
    right_pts = sorted([p.tolist() for p in scaled_runway_pts if p[0] > cx], key=lambda pt: pt[1])

    # Scale the centerline
    centerline_orig_coords = (np.array(centerline_model_coords) * [scale_x, scale_y]).astype(np.int32).tolist()

    # 6. SAVE RESULTS
    outdata = {
        "left_contour": left_pts,
        "right_contour": right_pts,
        "centerline": centerline_orig_coords
    }
    with open(os.path.join(OUT_DIR, "runway_detection.json"), "w") as f:
        json.dump(outdata, f, indent=2)

    # Draw visualization on the *original* BGR image
    vis = img_bgr.copy()
    if left_pts:
        cv2.polylines(vis, [np.array(left_pts)], isClosed=False, color=(0, 255, 0), thickness=3) # Green
    if right_pts:
        cv2.polylines(vis, [np.array(right_pts)], isClosed=False, color=(0, 0, 255), thickness=3) # Red
    if centerline_orig_coords:
        cv2.polylines(vis, [np.array(centerline_orig_coords)], isClosed=False, color=(255, 0, 0), thickness=3) # Blue

    cv2.imwrite(os.path.join(OUT_DIR, "runway_detection_result.png"), vis)
    print("Saved outputs to", OUT_DIR)

if __name__ == "__main__":
    predict()
    print("Done.")