import os
import shutil
import logging
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_yolo_dirs():
    """Create YOLO directory structure"""
    base_dirs = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val'
    ]
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Created directory: {d}")

def convert_to_yolo_format(bbox, img_width, img_height):
    """Convert bbox to YOLO format"""
    x_center = ((bbox[0] + bbox[2]) / 2) / img_width
    y_center = ((bbox[1] + bbox[3]) / 2) / img_height
    width = (bbox[2] - bbox[0]) / img_width
    height = (bbox[3] - bbox[1]) / img_height
    return [x_center, y_center, width, height]

def process_dataset():
    """Process and prepare dataset for YOLO"""
    source_dir = '../../antrenare/train_cnn'  # Updated path
    logging.info(f"Looking for dataset in: {os.path.abspath(source_dir)}")
    
    if not os.path.exists(source_dir):
        # Try alternative path
        source_dir = '../../antrenare/train_cnn'
        logging.info(f"Trying alternative path: {os.path.abspath(source_dir)}")
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory not found in either location")
    
    # Create clean dataset structure
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            dir_path = f'dataset/{subdir}/{split}'
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
            
    classes = ['dad', 'deedee', 'dexter', 'mom', 'unknown']
    
    # Process each class
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.exists(class_path):
            logging.error(f"Class directory not found: {class_path}")
            continue
            
        logging.info(f"Processing class: {class_name}")
        
        # Get all images for this class
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            logging.warning(f"No images found for class: {class_name}")
            continue
            
        # Split into train/val
        num_val = max(1, int(len(images) * 0.2))
        val_images = images[:num_val]
        train_images = images[num_val:]
        
        # Process train and val sets
        for img_name in train_images:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join('dataset/images/train', img_name)
            shutil.copy2(src_path, dst_path)
            
            # Create label file
            label_name = os.path.splitext(img_name)[0] + '.txt'
            with open(os.path.join('dataset/labels/train', label_name), 'w') as f:
                f.write(f"{classes.index(class_name)} 0.5 0.5 1.0 1.0\n")
                
        for img_name in val_images:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join('dataset/images/val', img_name)
            shutil.copy2(src_path, dst_path)
            
            # Create label file
            label_name = os.path.splitext(img_name)[0] + '.txt'
            with open(os.path.join('dataset/labels/val', label_name), 'w') as f:
                f.write(f"{classes.index(class_name)} 0.5 0.5 1.0 1.0\n")
                
        logging.info(f"Processed {len(train_images)} training and {len(val_images)} validation images for {class_name}")

if __name__ == '__main__':
    try:
        create_yolo_dirs()
        process_dataset()
    except Exception as e:
        logging.error(f"Dataset preparation failed: {e}")
