from ultralytics import YOLO
import cv2
from pathlib import Path
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load a pre-trained YOLOv8 model
    model = YOLO("yolov8m.pt")

    # Train the model
    model.train(
        data='../../training/YOLO/data.yaml',  # path to your data.yaml
        epochs=50,                             # total training epochs
        device=device,
        imgsz=640,                             # input image size
        batch=6,                              # batch size (reduce to 8 or 4 if out of memory)
        name='hexbug_yolo_m_run1',                  # folder name to save training results
        project='../../models/detection/YOLO',          # where to store results
        workers=0,
        auto_augment = 'randaugment',
        hsv_s = 0.5,
        hsv_v = 0.4,
        degrees=10,                          # Very gentle rotation
        translate=0.1,                     # Minimal translation
        scale=0.25,                          # Conservative scaling
        shear=10.0,                          # Reduced shear
        perspective=0.001,                 # Minimal perspective distortion
        flipud=0.0,                         # No vertical flip for tall frames
        fliplr=0.3,
    )

if __name__ == "__main__":
    main()