import os
import shutil
import logging
from collections import defaultdict
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_yolo_dirs():
    base_dirs = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val'
    ]
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Created directory: {d}")

def load_annotations(ann_file):
    annotations = defaultdict(list)
    with open(ann_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                img_name = parts[0]
                bbox = list(map(int, parts[1:5]))
                class_name = parts[5]
                annotations[img_name].append((bbox, class_name))
    return annotations

def convert_to_yolo_format(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [x_center, y_center, width, height]

def process_dataset():
    source_dir = '../../antrenare'
    ann_file = os.path.join(source_dir, 'toate_adnotarile.txt')
    
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotations file not found: {ann_file}")
    
    annotations = load_annotations(ann_file)
    logging.info(f"Loaded annotations for {len(annotations)} images")
    
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            dir_path = f'dataset/{subdir}/{split}'
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    
    classes = ['dad', 'deedee', 'dexter', 'mom', 'unknown']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    processed_count = 0
    for class_name in ['dad', 'deedee', 'dexter', 'mom']:
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.exists(class_dir):
            logging.error(f"Class directory not found: {class_dir}")
            continue
        
        images = sorted([f for f in os.listdir(class_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))])
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Process each split
        for split, img_list in [('train', train_images), ('val', val_images)]:
            for img_name in img_list:
                src_path = os.path.join(class_dir, img_name)
                ann_key = f"{img_name.split('.')[0]}_{class_name}.jpg"
                
                if ann_key not in annotations:
                    logging.warning(f"No annotations found for {ann_key}")
                    continue
                dst_img_path = os.path.join(f'dataset/images/{split}', img_name)
                shutil.copy2(src_path, dst_img_path)
                img = cv2.imread(src_path)
                if img is None:
                    logging.error(f"Could not read image: {src_path}")
                    continue
                height, width = img.shape[:2]
                
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(f'dataset/labels/{split}', label_name)
                
                with open(label_path, 'w') as f:
                    for bbox, cls_name in annotations[ann_key]:
                        if cls_name in class_to_idx:
                            yolo_bbox = convert_to_yolo_format(bbox, width, height)
                            cls_idx = class_to_idx[cls_name]
                            f.write(f"{cls_idx} {' '.join(map(str, yolo_bbox))}\n")
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logging.info(f"Processed {processed_count} images...")
    
    for ann_key, annotations_list in annotations.items():
        for bbox, cls_name in annotations_list:
            if cls_name in class_to_idx:
                yolo_bbox = convert_to_yolo_format(bbox, width, height)
                cls_idx = class_to_idx[cls_name]
                f.write(f"{cls_idx} {' '.join(map(str, yolo_bbox))}\n")
    
    logging.info(f"Dataset preparation completed! Processed {processed_count} images total")

if __name__ == '__main__':
    try:
        process_dataset()
    except Exception as e:
        logging.error(f"Dataset preparation failed: {e}")
