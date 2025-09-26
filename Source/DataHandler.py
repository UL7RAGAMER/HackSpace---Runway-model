import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
TARGET_SIZE = (1088, 1920)  
USE_LINES = True
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.Resize(height=320, width=320), 
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return albu.Compose(train_transform)


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_image_annotations(json_data):
    """
    Try to handle:
      - simple { "image_name.jpg": [ [x,y,...], ... ] }
      - COCO-like {"images": [...], "annotations": [...]}
    Returns dict: filename -> list of polygons (numpy int32 Nx2)
    """
    out = {}
    if isinstance(json_data, dict) and 'images' in json_data and 'annotations' in json_data:
        images = {img['id']: img['file_name'] for img in json_data['images']}
        for ann in json_data['annotations']:
            img_name = images.get(ann.get('image_id'))
            if img_name is None:
                continue
            segs = ann.get('segmentation', None)
            if not segs:
                continue
            # segmentation can be list of polygons (COCO)
            for poly in segs:
                coords = np.array(poly).reshape(-1, 2).astype(np.int32)
                out.setdefault(img_name, []).append(coords)
    else:
        # assume simple mapping: filename -> list-of-polygons
        for k, v in json_data.items():
            contours = []
            for poly in v:
                arr = np.array(poly)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 2)
                contours.append(arr.astype(np.int32))
            out[k] = contours
    return out

def polygons_to_mask_and_edge(polygons, h, w, edge_thickness=1):
    """
    polygons: list of numpy arrays (N_points, 2) in image coordinate space (x,y)
    returns: mask (H,W) uint8 {0,1}, edge (H,W) uint8 {0,1}
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygons:
        try:
            cv2.fillPoly(mask, pts=polygons, color=1)
        except Exception:
            # fallback: convert to list of lists
            pts = [p.reshape((-1,1,2)).astype(np.int32) for p in polygons]
            cv2.fillPoly(mask, pts=pts, color=1)

    edge = np.zeros((h, w), dtype=np.uint8)
    if polygons:
        for poly in polygons:
            try:
                cv2.polylines(edge, [poly], isClosed=True, color=1, thickness=edge_thickness, lineType=cv2.LINE_AA)
            except Exception:
                cv2.polylines(edge, [poly.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=1, thickness=edge_thickness)
    return mask, edge

# ----------------------------------------------------------------------
# Dataset: loads image, area mask image file, and builds edge from JSON polygons
# ----------------------------------------------------------------------
class CombinedRunwayDataset(Dataset):
    def __init__(self, images_dir, area_labels_dir, lines_json_path=None, augmentation=None, target_size=TARGET_SIZE, img_suffixes=('.jpg','.png','.jpeg')):
        self.images_dir = images_dir
        self.area_labels_dir = area_labels_dir
        self.augmentation = augmentation
        self.h, self.w = target_size
        self.img_suffixes = img_suffixes

        # list image files
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(img_suffixes)]
        self.image_files.sort()

        # read JSON polygons if provided
        self.ann_map = {}
        if lines_json_path and os.path.exists(lines_json_path):
            try:
                json_data = load_json(lines_json_path)
                self.ann_map = get_image_annotations(json_data)
            except Exception as e:
                print("Warning: failed to parse lines JSON:", e)
                self.ann_map = {}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Unable to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]

        # load area mask image (assumes filename matches)
        mask = None
        area_mask_path_candidate = os.path.join(self.area_labels_dir, fname)
        if not os.path.exists(area_mask_path_candidate):
            base = os.path.splitext(fname)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
                p = os.path.join(self.area_labels_dir, base + ext)
                if os.path.exists(p):
                    area_mask_path_candidate = p
                    break
        if os.path.exists(area_mask_path_candidate):
            mask_img = cv2.imread(area_mask_path_candidate, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise RuntimeError(f"Unable to read mask: {area_mask_path_candidate}")
            _, mask = cv2.threshold(mask_img, 127, 1, cv2.THRESH_BINARY)
            mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # polygons for edges from JSON map
        polygons = self.ann_map.get(fname, [])

        # scale polygons from original image size to target if needed
        scaled_polys = []
        if len(polygons) > 0:
            scale_x = float(self.w) / float(ow)
            scale_y = float(self.h) / float(oh)
            for p in polygons:
                poly = p.astype(np.float32).copy()
                poly[:, 0] *= scale_x
                poly[:, 1] *= scale_y
                scaled_polys.append(np.round(poly).astype(np.int32))

        edge = np.zeros((self.h, self.w), dtype=np.uint8)
        if scaled_polys:
            mask_from_polys, edge = polygons_to_mask_and_edge(scaled_polys, self.h, self.w, edge_thickness=1)
            # OPTIONAL: if you want to trust polygon fills too, combine them:
            # mask = np.maximum(mask, mask_from_polys)

        # Resize image to target (H,W) -> cv2 uses (W,H)
        img_resized = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # If augmentation is provided (albumentations), call it with both masks
        if self.augmentation:
            # Albumentations expects masks as a list in the same order we pass them
            # After ToTensorV2, augmentation may return torch.Tensor for image/masks
            aug = self.augmentation(image=img_resized, masks=[mask, edge])
            img_resized = aug['image']
            aug_masks = aug.get('masks', None)
            if aug_masks is not None and len(aug_masks) >= 2:
                mask, edge = aug_masks[0], aug_masks[1]

        # --- Helper conversions: accept either numpy arrays or torch tensors ---
        def img_to_tensor(x):
            # returns torch.FloatTensor (C,H,W) with values in [0,1]
            if torch.is_tensor(x):
                t = x.float()
                # If already CHW
                if t.dim() == 3 and t.shape[0] in (1, 3, 4):
                    return t
                # If HWC, convert to CHW
                if t.dim() == 3 and t.shape[2] in (1, 3, 4):
                    return t.permute(2, 0, 1)
                # Unexpected shapes: try flatten->float
                return t
            else:
                # numpy HWC expected
                arr = x
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                arr = arr / 255.0
                return torch.from_numpy(np.transpose(arr, (2, 0, 1)).astype(np.float32))

        def mask_to_tensor(m):
            # returns torch.FloatTensor (H,W) values 0/1
            if torch.is_tensor(m):
                t = m.float()
                # if channel-first single channel (1,H,W) -> squeeze
                if t.dim() == 3 and t.shape[0] == 1:
                    return t[0]
                # if HWC with single channel (H,W,1) -> squeeze last dim
                if t.dim() == 3 and t.shape[2] == 1:
                    return t.squeeze(2)
                # already H,W or other acceptable form
                if t.dim() == 2:
                    return t
                # fallback: try flattening first channel
                return t.squeeze()
            else:
                arr = m
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                return torch.from_numpy(arr)

        # Convert image and targets to tensors
        img_tensor = img_to_tensor(img_resized)            # (C,H,W) float32
        mask_t = mask_to_tensor(mask)                      # (H,W) float32
        edge_t = mask_to_tensor(edge)                      # (H,W) float32

        # Compose final target tensor
        if USE_LINES:
            target_tensor = torch.stack([mask_t, edge_t], dim=0)  # (2,H,W)
        else:
            target_tensor = mask_t.unsqueeze(0)                   # (1,H,W)

        # Ensure dtype float32
        img_tensor = img_tensor.to(torch.float32)
        target_tensor = target_tensor.to(torch.float32)

        return img_tensor, target_tensor




# ----------------------------------------------------------------------
# Model + loss
# ----------------------------------------------------------------------
def build_model(use_lines):
    classes = 2 if use_lines else 1
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=classes,
    )
    return model

def dice_loss_from_logits(logits, targets, eps=1e-7):
    """
    logits: (B, H, W)
    targets: (B, H, W) float 0/1
    calculates Dice loss between sigmoid(logits) and targets
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1,2)) + eps
    den = probs.sum(dim=(1,2)) + targets.sum(dim=(1,2)) + eps
    loss = 1.0 - (num / den)
    return loss.mean()

bce_loss_fn = torch.nn.BCEWithLogitsLoss()

def combined_loss_fn(outputs, targets, mask_weight=1.0, edge_weight=1.0):
    """
    outputs: (B, C, H, W)
    targets: (B, C, H, W)
    When USE_LINES True: C==2 (mask, edge)
    """
    if outputs.size(1) == 1:
        # single channel: treat as mask-only
        logits = outputs[:, 0, :, :]
        true = targets[:, 0, :, :]
        return dice_loss_from_logits(logits, true)
    else:
        mask_logits = outputs[:, 0, :, :]
        edge_logits = outputs[:, 1, :, :]
        mask_true = targets[:, 0, :, :]
        edge_true = targets[:, 1, :, :]

        l_mask = dice_loss_from_logits(mask_logits, mask_true)
        l_edge = bce_loss_fn(edge_logits, edge_true)
        return mask_weight * l_mask + edge_weight * l_edge

    

