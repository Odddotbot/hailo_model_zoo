import cv2
import os
from pathlib import Path


# Variables
src_val_dir = Path('/datasets/hailo_calibration/val/images')
tgt_val_dir = Path('/datasets/hailo_calibration/val/images_448')
img_filepaths = [img_path for img_path in sorted(os.listdir(src_val_dir)) if img_path.endswith('jpg')]
for img_filepath in img_filepaths:
    img = cv2.imread(src_val_dir / img_filepath)
    img = cv2.resize(img, (448, 448))
    cv2.imwrite(tgt_val_dir / img_filepath, img)