import os
import json
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from DataHandler import *
IMAGES_DIR = r"C:\Users\Siddharth Kumar\Desktop\Hackathon\Hackathon SDG\HackSpace---Runway-model\DataSets\1920x1080\1920x1080\train"
AREA_LABELS_DIR = r"C:\Users\Siddharth Kumar\Desktop\Hackathon\Hackathon SDG\HackSpace---Runway-model\DataSets\labels\labels\areas\train_labels_1920x1080"
LINES_JSON = r"C:\Users\Siddharth Kumar\Desktop\Hackathon\Hackathon SDG\HackSpace---Runway-model\DataSets\labels\labels\lines\train_labels_640x360.json"
BATCH_SIZE = 64  
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
def train():
    try:
        augmentation = get_training_augmentation()
    except Exception:
        augmentation = None
        print("Warning: get_training_augmentation not available or failed; continuing without augmentation.")

    dataset = CombinedRunwayDataset(IMAGES_DIR, AREA_LABELS_DIR, lines_json_path=LINES_JSON if USE_LINES else None, augmentation=augmentation, target_size=TARGET_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = build_model(USE_LINES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(DEVICE, dtype=torch.float32)
            targets = targets.to(DEVICE, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images)  # (B, C, H, W)
            loss_val = combined_loss_fn(outputs, targets, mask_weight=1.0, edge_weight=1.0)
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item() * images.size(0)


            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(loader)}]  Loss: {loss_val.item():.4f}")

        avg_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Avg Loss: {avg_loss:.4f}")

    save_name = "runway.pth" if USE_LINES else "runway_no_lines.pth"
    torch.save(model.state_dict(), save_name)
    print("Saved model to", save_name)

if __name__ == "__main__":
    train()
