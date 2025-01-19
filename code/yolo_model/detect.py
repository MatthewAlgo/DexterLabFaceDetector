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
    validation_dir = '../validare/validare'  # Updated path to validation directory
    
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
    
    # Process each validation image
    for img_name in sorted(os.listdir(validation_dir)):  # Sort for consistent processing
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
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

if __name__ == '__main__':
    run_detection()
