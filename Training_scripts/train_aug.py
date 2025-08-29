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
    model = YOLO("model_path/best.pt")

    model.train(
        data='../../training/YOLO/data.yaml',  # path to your data.yaml
        epochs=30,
        device=device,
        imgsz=640,                             # input image size 
        batch = 10,
        name = 'hexbug_yolo_m_polished',                  # folder name to save training results
        project='../../models/detection/YOLO',
         # ---------------- Augmentation ----------------
        mosaic=0,                   # turn off mosaic
        mixup=0,                    # turn off mix-up
        rect=True,                  # rectangular dataloader
        perspective=0.001,          # mild keystone/tilt
        degrees=5,
        shear=2,
        scale=0.2,
        translate=0.05,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.5,  
        flipud=0.0,
        fliplr=0.5,
        resume=True )



    # # Train the model (aug 1)
    # model.train(
    #     data='../../training/YOLO/data.yaml',  # path to your data.yaml
    #     epochs=30,                             # total training epochs
    #     device=device,
    #     imgsz=640,                             # input image size
    #     batch=16,                              # batch size (reduce to 8 or 4 if out of memory)
    #     workers = 0,
    #     name='hexbug_yolo_aug_run1',                  # folder name to save training results
    #     project='../../models/detection/YOLO',          # where to store results
    #     resume = False,
    #     auto_augment = 'randaugment',
    #     hsv_s = 0.5,
    #     hsv_v = 0.4,
    #     degrees= 10,
    #     shear= 3.0,
    #     scale= 0.5,
    #     perspective = 0.0005,
    # )

    # train aug 24 - Building on successful run1 with larger images
    # model.train(
    #     data = '../../training/YOLO/data.yaml',  # path to your data.yaml
    #     epochs = 80,                             # A bit more since run1 was still improving at epoch 50
    #     device = device,
    #     imgsz = 960,                             # increased input image for better small object detection
    #     batch = 8,                              # batch size (reduce to 8 or 4 if out of memory)
    #     workers = 0,
    #     lr0 = 0.001,                           # More conservative LR similar to successful run1
    #     lrf = 0.1,                             # Learning rate decay factor
    #     patience = 25,                          # Early stopping if no improvement for 25 epochs
    #     name = 'hexbug_yolo_aug_run2',           # folder name to save training results
    #     project = '../../models/detection/YOLO', # where to store results
    #     resume = False,
    #     hsv_h=0.05,      # colour-cast robustness
    #     hsv_s=0.8,       # "
    #     hsv_v=0.6,       # dark-frame robustness
    #     close_mosaic=10,  # close mosaic augmentation for better small object detection
    # )




    # model.train(
    #     data='../../training/Updated_YOLO/hexbugs.yaml',
    #     epochs=30,            # short, will stop early if flat
    #     device=device,
    #     imgsz=640,            # frees VRAM, steadier gradients
    #     batch=8,
    #     lr0=0.0015, lrf=0.1,  # gentler LR
    #     patience=10,           # stops if no val gain for 6 epochs
    #     hsv_h=0.2, hsv_s=0.4, hsv_v=0.4,
    #     mosaic=0.10,          # lighter mosaic
    #     close_mosaic=5,
    #     bgr=0.05,             # keep but halve probability
    #     name='hexbug_yolo_finetuned',
    #     project='../../models/detection/YOLO'
    # )

    # Current training - wide image approach
    # model.train(
    #     data='../../training/SkewedYOLO/hexbugs.yaml',
    #     epochs=35,            # short, will stop early if flat
    #     device=device,
    #     imgsz=1024,            # to handle wide images
    #     batch=4,
    #     optimizer='SGD',
    #     lr0=3e-4, lrf=0.033,      # hits 1e-5 at epoch 35
    #     warmup_epochs=2,
    #     mosaic=0.0,
    #     patience=10,           # stops if no val gain for 6 epochs
    #     name='hexbug_yolo_finetuned2',
    #     project='../../models/detection/YOLO'
    # )

    # Fine-tuning with cosine learning rate and adjusted loss weights
    # model.train(
    #     data='../../training/SkewedYOLO/hexbugs.yaml',
    #     epochs=35,            # short, will stop early if flat
    #     device=device,
    #     imgsz=1024,            # to handle wide images
    #     batch=4,
    #     optimizer='SGD',
    #     lr0=3e-4, lrf=0.033,
    #     cos_lr=True,               # <-- switch to cosine
    #     warmup_epochs=2,
    #     cls=0.8,                   # raise cls-loss weight (default 0.5)
    #     dropout=0.1, 
    #     warmup_epochs=2,
    #     mosaic=0.0,
    #     patience=10,           # stops if no val gain for 6 epochs
    #     name='hexbug_yolo_finetuned2',
    #     project='../../models/detection/YOLO'
    # )
 
    # New training config optimized for multiple small objects and varying conditions (try after current)
    # model.train(
    #     data= '../../training/YOLO/data.yaml',
    #     epochs=75,
    #     device=device,
    #     imgsz=1024,
    #     batch=4,                        # 5–6 if VRAM allows
    #     optimizer='AdamW',
    #     lr0=3e-4, lrf=0.033, cos_lr=True,
    #     warmup_epochs=2,
    #     patience=15,
    #     mosaic=0.0,
    #     hsv_h=0.3, hsv_s=0.7, hsv_v=0.5,
    #     degrees=15, translate=0.2, scale=0.4,
    #     shear=5.0, perspective=0.0015,
    #     flipud=0.1, fliplr=0.5,
    #     dropout=0.1,
    #     name='hexbug_multi_object_v1',
    #     project='../../models/detection/YOLO',
    #     save_period=10,
    #     val=True,
    # )

    # negative fine-tuning with adjusted parameters
    # model.train(
    #     data='../../training/YOLO/data.yaml',
    #     epochs=20,
    #     imgsz=1024,
    #     batch=6,
    #     optimizer='AdamW',
    #     lr0=2e-4, lrf=0.05, cos_lr=True,
    #     warmup_epochs=1,
    #     mosaic=0.0,
    #     hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
    #     degrees=12, translate=0.15, scale=0.35,
    #     shear=8.0, perspective=0.0008,
    #     flipud=0.1, fliplr=0.5,
    #     cls=0.7, box=5.0, dfl=1.5,
    #     patience=8, 
    #     name='hexbug_final',
    #     project='../../models/detection/YOLO',
    # )

    # model.train(
    #     data='../../training/YOLO/data.yaml',
    #     epochs=25,
    #     imgsz=1280,                         # Larger input size for small objects
    #     batch=4,                            # Reduced batch for larger images
    #     optimizer='AdamW',
    #     lr0=1e-4, lrf=0.01, cos_lr=True,   # Lower LR for fine-tuning
    #     warmup_epochs=2,
    #     mosaic=0.0,                         # Disable mosaic for small objects
    #     copy_paste=0.1,                     # Help with small object detection
    #     mixup=0.1,                          # Additional augmentation for robustness
    #     # Minimal geometric augmentations for tall frames
    #     degrees=5,                          # Very gentle rotation
    #     translate=0.05,                     # Minimal translation
    #     scale=0.1,                          # Conservative scaling
    #     shear=2.0,                          # Reduced shear
    #     perspective=0.001,                 # Minimal perspective distortion
    #     flipud=0.0,                         # No vertical flip for tall frames
    #     fliplr=0.3,                         # Moderate horizontal flip
    #     # Color augmentations for lighting variations
    #     hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
    #     # Loss weights optimized for small objects
    #     cls=1.0, box=7.5, dfl=2.0,         # Higher box loss for small objects
    #     patience=8,
    #     # Multi-scale training for various aspect ratios
    #     rect=True,                          # Rectangular training for varied aspect ratios
    #     name='hexbug_tall_frames_optimized',
    #     project='../../models/detection/YOLO',
    # )


#     model.train(
#         data='../../training/YOLO/data.yaml',
#         epochs=35,
#         imgsz=960,                         # Larger input size for small objects
#         batch=4,                            # Reduced batch for larger images
#         optimizer='AdamW',
#         lr0=1e-4, lrf=0.01, cos_lr=True,   # Lower LR for fine-tuning
#         warmup_epochs=2,
#         mosaic=0.0,                         # Disable mosaic for small objects
#         copy_paste=0.1,                     # Help with small object detection
#         mixup=0.1,                          # Additional augmentation for robustness
#         # Minimal geometric augmentations for tall frames
#         degrees=10,                          # Very gentle rotation
#         translate=0.1,                     # Minimal translation
#         scale=0.5,                          # Conservative scaling
#         shear=10.0,                          # Reduced shear
#         perspective=0.001,                 # Minimal perspective distortion
#         flipud=0.0,                         # No vertical flip for tall frames
#         fliplr=0.3,                         # Moderate horizontal flip
#         # Color augmentations for lighting variations
#         hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
#         # Loss weights optimized for small objects
#         cls=1.0, box=7.5, dfl=2.0,         # Higher box loss for small objects
#         patience=10,
#         # Multi-scale training for various aspect ratios
#         auto_augment = "randaugment",
#         name='hexbug_tall_frames_perspective',
#         project='../../models/detection/YOLO',

# )

if __name__ == "__main__":
    main()

