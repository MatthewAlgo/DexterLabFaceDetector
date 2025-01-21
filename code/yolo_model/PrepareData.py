import os
import shutil
import logging
from collections import defaultdict
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_yolo_directories():
    # Foldere pentru yolo
    yolo_dataset_directories = ['dataset/images/train', 'dataset/images/val', 'dataset/labels/train', 'dataset/labels/val']
    for d in yolo_dataset_directories:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Created directory: {d}")

def parse_annotations(ann_file):
    image_annotations = defaultdict(list)
    with open(ann_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                image_filename = parts[0]
                bounding_coordinates = list(map(int, parts[1:5]))
                class_name = parts[5]
                image_annotations[image_filename].append((bounding_coordinates, class_name))
    return image_annotations

def convert_bbox_to_yolo_format(bounding_box, img_width, img_height):
    bounding_box_x1, bounding_box_top, bounding_box_x2, bounding_box_bottom = bounding_box
    x_center = ((bounding_box_x1 + bounding_box_x2) / 2) / img_width
    y_center = ((bounding_box_top + bounding_box_bottom) / 2) / img_height
    width = (bounding_box_x2 - bounding_box_x1) / img_width
    height = (bounding_box_bottom - bounding_box_top) / img_height
    return [x_center, y_center, width, height]

def prepare_image_dataset():
    annotations_source_dir = '../../antrenare'
    annotations_file_path = os.path.join(annotations_source_dir, 'toate_adnotarile.txt')
    
    if not os.path.exists(annotations_file_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file_path}")
    annotations = parse_annotations(annotations_file_path)
    logging.info(f"Loaded annotations for {len(annotations)} images")
    
    for data_split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            dir_path = f'dataset/{subdir}/{data_split}'
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    
    classes = ['dad', 'deedee', 'dexter', 'mom', 'unknown']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    processed_count = 0
    
    for class_name in ['dad', 'deedee', 'dexter', 'mom']:
        class_image_directory = os.path.join(annotations_source_dir, class_name)
        if not os.path.exists(class_image_directory):
            logging.error(f"Class directory not found: {class_image_directory}")
            continue
        
        sorted_image_filenames = sorted([f for f in os.listdir(class_image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))])
        training_split_index = int(len(sorted_image_filenames) * 0.8)
        train_image_list = sorted_image_filenames[:training_split_index]
        validation_image_list = sorted_image_filenames[training_split_index:]
        
        # Process each split
        for data_split, image_train_list in [('train', train_image_list), ('val', validation_image_list)]:
            for img_name in image_train_list:
                source_image_path = os.path.join(class_image_directory, img_name)
                image_annotation_id = f"{img_name.split('.')[0]}_{class_name}.jpg"
                if image_annotation_id not in annotations:
                    logging.warning(f"No annotations found for {image_annotation_id}")
                    continue
                
                destination_image_path = os.path.join(f'dataset/images/{data_split}', img_name)
                shutil.copy2(source_image_path, destination_image_path)
                img = cv2.imread(source_image_path)
                if img is None:
                    logging.error(f"Could not read image: {source_image_path}")
                    continue
                # Luam dimensiunile imaginii
                height, width = img.shape[:2]
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(f'dataset/labels/{data_split}', label_name)
                
                with open(label_path, 'w') as f:
                    for object_bounding_box, cls_name in annotations[image_annotation_id]:
                        if cls_name in class_to_idx:
                            yolo_bounding_box = convert_bbox_to_yolo_format(object_bounding_box, width, height)
                            class_index = class_to_idx[cls_name]
                            f.write(f"{class_index} {' '.join(map(str, yolo_bounding_box))}\n")
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logging.info(f"Processed {processed_count} images...")
                    
    # Luam fiecare imagine si adnotarea ei si le scriem in fisierul de adnotari
    for image_annotation_id, annotations_list in annotations.items():
        for object_bounding_box, cls_name in annotations_list:
            if cls_name in class_to_idx:
                yolo_bounding_box = convert_bbox_to_yolo_format(object_bounding_box, width, height)
                class_index = class_to_idx[cls_name]
                f.write(f"{class_index} {' '.join(map(str, yolo_bounding_box))}\n")
    
    logging.info(f"Dataset preparation completed! Processed {processed_count} images total")

if __name__ == '__main__':
    try:
        prepare_image_dataset()
    except Exception as e:
        logging.error(f"Dataset preparation failed: {e}")
