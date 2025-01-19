import os
import yaml
import torch
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'runs',
        'detections/task1',
        'detections/task2',
        'detections/detections',
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Created directory: {d}")

def train_yolo():
    """Train YOLO model on Dexter dataset"""
    try:
        # Initialize model
        logging.info("Initializing YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        
        # Get absolute path to dataset.yaml
        dataset_yaml = os.path.abspath('dataset.yaml')
        logging.info(f"Using dataset config: {dataset_yaml}")
        
        logging.info("Starting training...")
        results = model.train(
            data=dataset_yaml,  # Use absolute path
            epochs=50,
            imgsz=640,
            batch=16,
            name='dexter_faces',
            verbose=True,
            device='0' if torch.cuda.is_available() else 'cpu',
            save=True,
            patience=15
        )
        
        logging.info("Training completed successfully!")
        return model
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    try:
        setup_directories()
        logging.info("Checking dataset structure...")
        
        # Verify dataset exists
        if not os.path.exists('dataset/images/train') or \
           not os.path.exists('dataset/images/val'):
            logging.error("Dataset not prepared! Please run prepare_data.py first")
            exit(1)
            
        # Check if there are any images
        train_images = os.listdir('dataset/images/train')
        val_images = os.listdir('dataset/images/val')
        
        if not train_images or not val_images:
            logging.error("No images found in dataset! Please run prepare_data.py first")
            exit(1)
        
        logging.info(f"Found {len(train_images)} training images and {len(val_images)} validation images")
        model = train_yolo()
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
