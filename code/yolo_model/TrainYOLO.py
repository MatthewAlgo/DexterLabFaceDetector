import os
import yaml
import torch
from ultralytics import YOLO
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_training_directories():
    # Foldere de antrenare si totul necesar
    training_directory_paths = [
        'runs', 'detections/task1', 'detections/task2', 'detections/detections',
        'dataset/images/train', 'dataset/labels/train',
        'dataset/images/val', 'dataset/labels/val'
    ]
    for d in training_directory_paths:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Created directory: {d}")

def run_yolo_training():
    try:
        logging.info("Initializing YOLOv8 model...")
        trained_yolo_model = YOLO('yolov8n.pt')
        yaml_dataset_file = os.path.abspath('dataset.yaml')
        execution_device = '0' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using dataset config: {yaml_dataset_file}")
        
        logging.info("Starting training...")
        trained_yolo_model.train(
            data=yaml_dataset_file, epochs=50, imgsz=512, batch=16, name='dexter_faces',
            verbose=True, device=execution_device, save=True, patience=15
        )
        
        logging.info("Training completed successfully!")
        return trained_yolo_model
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    try:
        initialize_training_directories()
        logging.info("Checking dataset structure...")
        if not os.path.exists('dataset/images/train') or \
           not os.path.exists('dataset/images/val'):
            logging.error("Dataset not prepared! Please run PrepareData.py first")
            exit(1)
        # Verificam structura datasetului
        training_image_list = os.listdir('dataset/images/train')
        validation_images = os.listdir('dataset/images/val')
        if not training_image_list or not validation_images:
            logging.error("No images found in dataset! Please run PrepareData.py first")
            exit(1)
        
        logging.info(f"Found {len(training_image_list)} training images and {len(validation_images)} validation images")
        trained_yolo_model = run_yolo_training()
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
