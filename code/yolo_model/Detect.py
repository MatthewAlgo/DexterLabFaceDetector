import os
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import torch

class Dexter_YOLO_Detector:
    def __init__(self):
        self.face_detection_model = YOLO('runs/detect/dexter_faces/weights/best.pt')
        self.face_classes = ['dad', 'deedee', 'dexter', 'mom', 'unknown']
        
    def perform_face_detection(self, image_path):
        detection_results = self.face_detection_model(image_path)
        return detection_results[0]
    
    def extract_detections(self, results, image):
        # Pentru imaginea curenta, extrage bounding boxes, scoruri si clasele 
        bounding_boxes = results.boxes.cpu().numpy()
        detected_bounding_boxes = []
        detection_confidences = []
        detected_classes = []
        
        for box in bounding_boxes:
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            detected_bounding_boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
            detection_confidences.append(confidence)
            detected_classes.append(self.face_classes[class_id])
            
        return np.array(detected_bounding_boxes), np.array(detection_confidences), detected_classes

def perform_face_detection():
    face_detection_model = Dexter_YOLO_Detector()
    validation_directory = '../../validare/validare' 
    if not os.path.exists(validation_directory):
        raise FileNotFoundError(f"Validation directory not found at {validation_directory}")
    face_detections_list = []
    detection_scores = []
    detection_file_names = []
    # Output dirs
    os.makedirs('detections/task1', exist_ok=True)
    os.makedirs('detections/detections', exist_ok=True)

    if os.path.exists('detections/task1/task1_yolo_detections.txt'):
        os.remove('detections/task1/task1_yolo_detections.txt')
    
    character_detection_results = {
        'dad': {'detections': [], 'scores': [], 'file_names': []},
        'deedee': {'detections': [], 'scores': [], 'file_names': []},
        'dexter': {'detections': [], 'scores': [], 'file_names': []},
        'mom': {'detections': [], 'scores': [], 'file_names': []},
    }
    
    os.makedirs('detections/task2', exist_ok=True)
    
    detected_faces_task1 = []
    detected_faces_task1_scores = []
    task1_image_filenames = []
    
    for validation_image_name in sorted(os.listdir(validation_directory)):  # Sort for consistent processing
        if not validation_image_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # '001.jpg' instead of 'image_001.jpg'
        base_name = validation_image_name.split('_')[-1] if '_' in validation_image_name else validation_image_name
        
        image_path = os.path.join(validation_directory, validation_image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        detection_results = face_detection_model.perform_face_detection(image_path)
        detected_faces, face_detection_scores, recognized_classes = face_detection_model.extract_detections(detection_results, image)
        
        # Save visualization
        output_image_with_annotations = image.copy()
        for det, score, cls_name in zip(detected_faces, face_detection_scores, recognized_classes):
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, det)
            cv2.rectangle(output_image_with_annotations, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 3)
            detection_annotation = f"{cls_name} {score:.2f}"
            
            (w, h), _ = cv2.getTextSize(detection_annotation, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output_image_with_annotations, (top_left_x, top_left_y-h-10), (top_left_x+w+10, top_left_y), (0, 0, 255), -1)
            cv2.putText(output_image_with_annotations, detection_annotation, (top_left_x+5, top_left_y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(f'detections/detections/{validation_image_name}', output_image_with_annotations)
        
        # Store in file for task1
        if len(detected_faces) > 0:
            face_detections_list.extend(detected_faces)
            detection_scores.extend(face_detection_scores)
            detection_file_names.extend([validation_image_name] * len(detected_faces))
            for det, score, cls_name in zip(detected_faces, face_detection_scores, recognized_classes):
                if cls_name in character_detection_results:
                    character_detection_results[cls_name]['detections'].append(det)
                    character_detection_results[cls_name]['scores'].append(score)
                    character_detection_results[cls_name]['file_names'].append(validation_image_name)
            
            # Write to task1 results file
            with open('detections/task1/task1_yolo_detections.txt', 'a') as f:
                for det, score, cls_name in zip(detected_faces, face_detection_scores, recognized_classes):
                    top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, det)
                    f.write(f"{validation_image_name} {top_left_x} {top_left_y} {bottom_right_x} {bottom_right_y} {cls_name}\n")
    
    # Pentru task 1, toate npy-urile
    if face_detections_list:
        np.save('detections/task1/detections_all_faces.npy', np.array(face_detections_list))
        np.save('detections/task1/scores_all_faces.npy', np.array(detection_scores))
        np.save('detections/task1/file_names_all_faces.npy', np.array(detection_file_names))
        
        print(f"Processing complete. Found {len(face_detections_list)} faces in {len(set(detection_file_names))} images.")
    else:
        print("No detections found in validation images.")
    
    # Save task2 for each character
    for char in character_detection_results:
        if character_detection_results[char]['detections']: 
            detection_output_path = f'detections/task2/detections_{char}.npy'
            detection_scores_file_path = f'detections/task2/scores_{char}.npy'
            detection_file_output_path = f'detections/task2/file_names_{char}.npy'
            
            np.save(detection_output_path, np.array(character_detection_results[char]['detections']))
            np.save(detection_scores_file_path, np.array(character_detection_results[char]['scores']))
            np.save(detection_file_output_path, np.array(character_detection_results[char]['file_names']))
            print(f"Saved {len(character_detection_results[char]['detections'])} detections for {char}")
    
    print("\nProcessing complete!")
    print(f"Total detections: {len(face_detections_list)} in {len(set(detection_file_names))} images")
    print("\nDetections by character:")
    for char in character_detection_results:
        print(f"{char}: {len(character_detection_results[char]['detections'])} detections")

    # Save task1 arrays
    if detected_faces_task1:
        detected_faces_task1 = np.array(detected_faces_task1)
        detected_faces_task1_scores = np.array(detected_faces_task1_scores)
        task1_image_filenames = np.array(task1_image_filenames)
        
        np.save('detections/task1/detections_all_faces.npy', detected_faces_task1)
        np.save('detections/task1/scores_all_faces.npy', detected_faces_task1_scores)
        np.save('detections/task1/file_names_all_faces.npy', task1_image_filenames)
        print(f"Task1: Saved {len(detected_faces_task1)} total detections")
    
    

if __name__ == '__main__':
    perform_face_detection()
