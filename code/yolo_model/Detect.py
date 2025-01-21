import os
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import torch

class Dexter_YOLO_Detector:
    def __init__(self):
        self.model = YOLO('runs/detect/dexter_faces/weights/best.pt')
        self.classes = ['dad', 'deedee', 'dexter', 'mom', 'unknown']
        
    def detect(self, image_path):
        """Run detection on an image"""
        results = self.model(image_path)
        return results[0]
    
    def process_results(self, results, image):
        """Process YOLO results and return structured data"""
        boxes = results.boxes.cpu().numpy()
        detections = []
        scores = []
        classifications = []
        
        for box in boxes:
            # Get bbox coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            detections.append([x1, y1, x2, y2])
            scores.append(confidence)
            classifications.append(self.classes[class_id])
            
        return np.array(detections), np.array(scores), classifications

def run_detection():
    """Run detection on validation set and save results"""
    detector = Dexter_YOLO_Detector()
    validation_dir = '../../validare/validare'  # Updated path to validation directory
    
    # Verify validation directory exists
    if not os.path.exists(validation_dir):
        raise FileNotFoundError(f"Validation directory not found at {validation_dir}")
    
    all_detections = []
    all_scores = []
    all_file_names = []
    
    # Create output directories if they don't exist
    os.makedirs('detections/task1', exist_ok=True)
    os.makedirs('detections/detections', exist_ok=True)
    
    # Clear previous results
    if os.path.exists('detections/task1/task1_yolo_detections.txt'):
        os.remove('detections/task1/task1_yolo_detections.txt')
    
    # Initialize character-specific storage
    character_detections = {
        'dad': {'detections': [], 'scores': [], 'file_names': []},
        'deedee': {'detections': [], 'scores': [], 'file_names': []},
        'dexter': {'detections': [], 'scores': [], 'file_names': []},
        'mom': {'detections': [], 'scores': [], 'file_names': []},
    }
    
    # Create task2 directory
    os.makedirs('detections/task2', exist_ok=True)
    
    # Initialize task-specific arrays
    task1_detections = []
    task1_scores = []
    task1_file_names = []
    
    # Initialize character-specific storage with numpy arrays
    character_detections = {
        'dad': {'detections': [], 'scores': [], 'file_names': []},
        'deedee': {'detections': [], 'scores': [], 'file_names': []},
        'dexter': {'detections': [], 'scores': [], 'file_names': []},
        'mom': {'detections': [], 'scores': [], 'file_names': []}
    }
    
    # Process each validation image
    for img_name in sorted(os.listdir(validation_dir)):  # Sort for consistent processing
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Get base filename without prefix (e.g., '001.jpg' instead of 'image_001.jpg')
        base_name = img_name.split('_')[-1] if '_' in img_name else img_name
        
        image_path = os.path.join(validation_dir, img_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Run detection
        results = detector.detect(image_path)
        detections, scores, classes = detector.process_results(results, image)
        
        # Save visualization
        viz_img = image.copy()
        for det, score, cls_name in zip(detections, scores, classes):
            x1, y1, x2, y2 = map(int, det)
            
            # Draw box with thicker lines for better visibility
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Add label with better visibility
            label = f"{cls_name} {score:.2f}"
            # Add background to text for better readability
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(viz_img, (x1, y1-h-10), (x1+w+10, y1), (0, 0, 255), -1)
            cv2.putText(viz_img, label, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(f'detections/detections/{img_name}', viz_img)
        
        # Store results
        if len(detections) > 0:
            all_detections.extend(detections)
            all_scores.extend(scores)
            all_file_names.extend([img_name] * len(detections))
            
            # Store character-specific results
            for det, score, cls_name in zip(detections, scores, classes):
                if cls_name in character_detections:  # Will now include 'unknown'
                    character_detections[cls_name]['detections'].append(det)
                    character_detections[cls_name]['scores'].append(score)
                    character_detections[cls_name]['file_names'].append(img_name)
            
            # Write to task1 results file
            with open('detections/task1/task1_yolo_detections.txt', 'a') as f:
                for det, score, cls_name in zip(detections, scores, classes):
                    x1, y1, x2, y2 = map(int, det)
                    f.write(f"{img_name} {x1} {y1} {x2} {y2} {cls_name}\n")
    
    # Save numpy arrays
    if all_detections:  # Only save if we have detections
        np.save('detections/task1/detections_all_faces.npy', np.array(all_detections))
        np.save('detections/task1/scores_all_faces.npy', np.array(all_scores))
        np.save('detections/task1/file_names_all_faces.npy', np.array(all_file_names))
        
        print(f"Processing complete. Found {len(all_detections)} faces in {len(set(all_file_names))} images.")
    else:
        print("No detections found in validation images.")
    
    # Save task2 arrays for each character
    for char in character_detections:
        if character_detections[char]['detections']:  # Only save if we have detections
            det_path = f'detections/task2/detections_{char}.npy'
            scores_path = f'detections/task2/scores_{char}.npy'
            files_path = f'detections/task2/file_names_{char}.npy'
            
            np.save(det_path, np.array(character_detections[char]['detections']))
            np.save(scores_path, np.array(character_detections[char]['scores']))
            np.save(files_path, np.array(character_detections[char]['file_names']))
            
            print(f"Saved {len(character_detections[char]['detections'])} detections for {char}")
    
    print("\nProcessing complete!")
    print(f"Total detections: {len(all_detections)} in {len(set(all_file_names))} images")
    print("\nDetections by character:")
    for char in character_detections:
        print(f"{char}: {len(character_detections[char]['detections'])} detections")

    # Save task1 arrays
    if task1_detections:
        task1_detections = np.array(task1_detections)
        task1_scores = np.array(task1_scores)
        task1_file_names = np.array(task1_file_names)
        
        np.save('detections/task1/detections_all_faces.npy', task1_detections)
        np.save('detections/task1/scores_all_faces.npy', task1_scores)
        np.save('detections/task1/file_names_all_faces.npy', task1_file_names)
        
        print(f"Task1: Saved {len(task1_detections)} total detections")
    
    # Save task2 arrays with proper numpy array conversion
    for char in character_detections:
        if character_detections[char]['detections']:
            # Convert lists to numpy arrays
            char_detections = np.array(character_detections[char]['detections'])
            char_scores = np.array(character_detections[char]['scores'])
            char_files = np.array(character_detections[char]['file_names'])
            
            # Ensure arrays are properly shaped
            if len(char_detections.shape) == 1:
                char_detections = char_detections.reshape(-1, 4)
            
            # Save arrays
            np.save(f'detections/task2/detections_{char}.npy', char_detections)
            np.save(f'detections/task2/scores_{char}.npy', char_scores)
            np.save(f'detections/task2/file_names_{char}.npy', char_files)
            
            print(f"Task2: Saved {len(char_detections)} detections for {char}")
            print(f"  Shapes: detections {char_detections.shape}, scores {char_scores.shape}, files {char_files.shape}")

    for char in character_detections:
        if character_detections[char]['detections']:
            detections = np.array(character_detections[char]['detections'], dtype=np.int32)
            scores = np.array(character_detections[char]['scores'], dtype=np.float32)
            file_names = np.array(character_detections[char]['file_names'])
            
            if len(detections.shape) == 1:
                detections = detections.reshape(-1, 4)
            
            np.save(f'detections/task2/detections_{char}.npy', detections)
            np.save(f'detections/task2/scores_{char}.npy', scores)
            np.save(f'detections/task2/file_names_{char}.npy', file_names)
            
            print(f"\nTask2 {char} arrays saved:")
            print(f"  detections shape: {detections.shape}")
            print(f"  scores shape: {scores.shape}")
            print(f"  file_names shape: {file_names.shape}")

if __name__ == '__main__':
    run_detection()
