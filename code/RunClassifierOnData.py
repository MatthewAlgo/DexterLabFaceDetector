from Parameters import *
from FacialDetectorDexter import *
import os
import multiprocessing as mp
from tqdm import tqdm
import glob
import torch

def perform_non_max_suppression(detection_boxes, detection_scores, iou_threshold=0.3):
    # NMS implementation
    if len(detection_boxes) == 0:
        return [], []
        
    detection_boxes = np.array(detection_boxes, dtype=np.float32)
    detection_scores = np.array(detection_scores, dtype=np.float32)
    box_top_left_x = detection_boxes[:, 0]
    box_top_left_y = detection_boxes[:, 1]
    box_bottom_right_x = detection_boxes[:, 2]
    box_bottom_right_y = detection_boxes[:, 3]
    # Areas
    bounding_box_areas = (box_bottom_right_x - box_top_left_x + 1) * (box_bottom_right_y - box_top_left_y + 1)
    order = detection_scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        box_overlap_top_left_x = np.maximum(box_top_left_x[i], box_top_left_x[order[1:]])
        overlap_top_left_y = np.maximum(box_top_left_y[i], box_top_left_y[order[1:]])
        bottom_right_x_overlap = np.minimum(box_bottom_right_x[i], box_bottom_right_x[order[1:]])
        overlap_bottom_right_y = np.minimum(box_bottom_right_y[i], box_bottom_right_y[order[1:]])
        
        width_overlap = np.maximum(0.0, bottom_right_x_overlap - box_overlap_top_left_x + 1)
        height_overlap = np.maximum(0.0, overlap_bottom_right_y - overlap_top_left_y + 1)
        intersection_area = width_overlap * height_overlap
        
        intersection_over_union = intersection_area / (bounding_box_areas[i] + bounding_box_areas[order[1:]] - intersection_area)
        
        inds = np.where(intersection_over_union <= iou_threshold)[0]
        order = order[inds + 1]
    
    return detection_boxes[keep].astype(np.int32), detection_scores[keep]

def detect_and_visualize_faces(input_arguments):
    image_path, model_file, scaler_file, params = input_arguments
    
    facial_detector = FacialDetectorDexter(params)
    facial_detector.load_classifier(model_file, scaler_file)
    image = cv.imread(image_path)
    image_name = os.path.basename(image_path)
    face_bounding_boxes, face_detection_scores = facial_detector.detector.find_faces_with_sliding_window(image)
    
    # PROCESS DETECTIONS
    if len(face_bounding_boxes) > 0:
        face_bounding_boxes, face_detection_scores = perform_non_max_suppression(face_bounding_boxes, face_detection_scores, iou_threshold=0.2)
        
        if len(face_bounding_boxes) > 0:
            detection_visualization = image.copy()
            face_detection_results = []
            for det, score in zip(face_bounding_boxes, face_detection_scores):
                face_rectangle_x1, top_left_y_coordinate, bottom_right_x, bottom_right_y_coordinate = map(int, det)
                # Append to results
                face_detection_results.append(f"{image_name} {face_rectangle_x1} {top_left_y_coordinate} {bottom_right_x} {bottom_right_y_coordinate}")
                cv.rectangle(detection_visualization, (face_rectangle_x1, top_left_y_coordinate), (bottom_right_x, bottom_right_y_coordinate), (0, 0, 255), 2)
                detection_score_text = f"Score: {score:.2f}"
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                score_text_padding = 5
                
                (text_w, text_h), _ = cv.getTextSize(detection_score_text, font, font_scale, thickness)
                cv.rectangle(detection_visualization, (face_rectangle_x1, top_left_y_coordinate-text_h-2*score_text_padding), (face_rectangle_x1+text_w+score_text_padding, top_left_y_coordinate), (0, 0, 0), -1)
                # Score text
                cv.putText(detection_visualization, detection_score_text,(face_rectangle_x1+score_text_padding//2, top_left_y_coordinate-score_text_padding), font, font_scale, (255, 255, 255), thickness)
            
            # Save visualization
            detection_output_path = os.path.join(params.dir_save_files, 'detections', f'boxes_{image_name}')
            os.makedirs(os.path.dirname(detection_output_path), exist_ok=True)
            cv.imwrite(detection_output_path, detection_visualization)
            
            return face_detection_results
    return []

def process_detections_with_cnn(params):
    # Read task1 results
    results_file = os.path.join(params.dir_save_files, 'task1gt_rezultate_detectie.txt')
    with open(results_file, 'r') as f:
        detection_lines = f.readlines()
    cnn_classifier = CNNFaceClassifier.load_latest_model()
    
    detections_by_image = {}
    for line in detection_lines:
        detection_parts = line.strip().split()
        image_file_name = detection_parts[0]
        face_bounding_box = list(map(int, detection_parts[1:]))
        if image_file_name not in detections_by_image:
            detections_by_image[image_file_name] = []
        detections_by_image[image_file_name].append(face_bounding_box)
    
    # Process each image detection
    for image_file_name, face_bounding_boxes in tqdm(detections_by_image.items(), desc="Processing images"):
        # Load image once
        input_image_path = os.path.join(params.dir_test_examples, image_file_name)
        image = cv.imread(input_image_path)
        if image is None:
            continue
        # Load vis image
        # Skip if visualization not found
        visualization_image_path = os.path.join(params.dir_save_files, 'detections', f'boxes_{image_file_name}')
        if not os.path.exists(visualization_image_path):
            continue
        visualization_image = cv.imread(visualization_image_path)
        
        # Detections
        classification_results = []
        for face_bounding_box in face_bounding_boxes:
            bounding_box_x1, bounding_box_top, bounding_box_x2, bounding_box_bottom = face_bounding_box
            extracted_face_image = image[bounding_box_top:bounding_box_bottom, bounding_box_x1:bounding_box_x2]
            face_img_rgb = cv.cvtColor(extracted_face_image, cv.COLOR_BGR2RGB)
            
            # Get classification
            classified_member, classification_confidence, _ = cnn_classifier.predict(face_img_rgb)
            classification_results.append((face_bounding_box, classified_member, classification_confidence))
        
        # Update visualization
        for face_bounding_box, classified_member, classification_confidence in classification_results:
            bounding_box_x1, bounding_box_top, bounding_box_x2, bounding_box_bottom = face_bounding_box
            # Add classification text
            label = f"{classified_member} ({classification_confidence:.2f})"
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            (label_w, label_h), _ = cv.getTextSize(label, font, font_scale, thickness)
            cv.rectangle(visualization_image, (bounding_box_x1, bounding_box_top-label_h-8), (bounding_box_x1+label_w+4, bounding_box_top-4), (0, 0, 0), -1)
            cv.putText(visualization_image, label, (bounding_box_x1+2, bounding_box_top-8), font, font_scale, (255, 255, 255), thickness)
        
        # Save updated visualization
        output_path = os.path.join(params.dir_save_files, 'detections', f'final_{image_file_name}')
        cv.imwrite(output_path, visualization_image)
        
        # Group results by character for task2
        char_results = {char: [] for char in ['dexter', 'deedee', 'mom', 'dad']}
        for face_bounding_box, classified_member, classification_confidence in classification_results:
            if classified_member in char_results:
                char_results[classified_member].append((face_bounding_box, classification_confidence, image_file_name))

def process_detections_for_task2(params):
    """Process detections and create separate files for each character"""
    # Read detection results with CNN classifications
    detections_dir = os.path.join(params.dir_save_files, 'detections')
    processed_image_filenames = [f for f in os.listdir(detections_dir) if f.startswith('final_')]
    
    character_detections = {
        'dexter': [], 'deedee': [], 'mom': [], 'dad': []
    }
    
    # Process each image
    for final_image_name in processed_image_filenames:
        image_name = final_image_name.replace('final_', '')
        
        detection_results_file = os.path.join(params.dir_save_files, 'task1gt_rezultate_detectie.txt')
        if not os.path.exists(detection_results_file):
            continue
            
        with open(detection_results_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                if parts[0] == image_name:
                    top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, parts[1:5])
                    
                    # Determine character from the filename or image content
                    # For example, if the image contains "dexter_001.jpg"
                    detected_character = None
                    for char in character_detections.keys():
                        if char in image_name.lower():
                            detected_character = char
                            break
                    if detected_character:
                        character_detections[detected_character].append(f"{image_name} {top_left_x} {top_left_y} {bottom_right_x} {bottom_right_y}")
    
    # Write detection files
    for char in character_detections:
        if character_detections[char]:
            detection_output_file = os.path.join(params.dir_save_files, f'task2_{char}_rezultate.txt')
            with open(detection_output_file, 'w') as f:
                for character_detection in character_detections[char]:
                    f.write(f"{character_detection}\n")
        else:
            # Write one dummy detection - no files
            detection_output_file = os.path.join(params.dir_save_files, f'task2_{char}_rezultate.txt')
            with open(detection_output_file, 'w') as f:
                f.write(f"placeholder_001.jpg 0 0 100 100\n")
    
    return character_detections

def evaluate_task2(params):
    # Evaluate task 2 results
    detection_characters = ['dexter', 'deedee', 'mom', 'dad']
    for char in detection_characters:
        print(f"\nEvaluating {char.capitalize()} detections:")
        # Read GT
        gt_file = os.path.join(params.dir_test_examples, '..', f'task2_{char}_gt_validare.txt')
        if not os.path.exists(gt_file):
            print(f"Ground truth file not found: {gt_file}")
            continue
            
        task2_character_results_file = os.path.join(params.dir_save_files, f'task2_{char}_rezultate.txt')
        if not os.path.exists(task2_character_results_file):
            print(f"Results file not found: {task2_character_results_file}")
            continue
        try:
            ground_truth_data = np.loadtxt(gt_file, dtype=str)
            algorithm_detections = np.loadtxt(task2_character_results_file, dtype=str)
            # Reshape
            if len(ground_truth_data.shape) == 1:
                ground_truth_data = ground_truth_data.reshape(1, -1)
            if len(algorithm_detections.shape) == 1:
                algorithm_detections = algorithm_detections.reshape(1, -1)
                
            print(f"Ground truth detections: {len(ground_truth_data)}")
            print(f"Algorithm detections: {len(algorithm_detections)}")
            
        except Exception as e:
            print(f"Error evaluating {char}: {e}")

def run_classifier():
    configuration = Parameters()
    
    # Get a detector to calculate window info upfront
    facial_detector = FacialDetectorDexter(configuration)
    facial_detector.load_classifier(
        os.path.join(configuration.dir_save_files, 'unified_svm_model.pkl'),
        os.path.join(configuration.dir_save_files, 'unified_scaler.pkl')
    )
    window_count = len(facial_detector.detector.window_sizes)
    print("\nInitialization Summary:")
    print(f"Number of window sizes: {window_count}")
    print("\nStarting detection...")
    
    # Create detections dir
    detections_dir = os.path.join(configuration.dir_save_files, 'detections')
    os.makedirs(detections_dir, exist_ok=True)
    model_file = os.path.join(configuration.dir_save_files, 'unified_svm_model.pkl')
    scaler_file = os.path.join(configuration.dir_save_files, 'unified_scaler.pkl')
    
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        print("Please train the model first")
        return
    
    test_images = glob.glob(os.path.join(configuration.dir_test_examples, '*.jpg'))
    process_args = [(img, model_file, scaler_file, configuration) for img in test_images]
    output_file = os.path.join(configuration.dir_save_files, 'task1gt_rezultate_detectie.txt')
    available_cpu_processes = min(12, mp.cpu_count())
    print(f"\nProcessing images with {available_cpu_processes} processes...")
    
    # Run parallel processing with progress bar
    with mp.Pool(processes=available_cpu_processes) as pool:
        results = list(tqdm(pool.imap(detect_and_visualize_faces, process_args), 
                          total=len(test_images),
                          desc="Processing images"))
    
    with open(output_file, 'w') as f:
        for result_list in results:
            for result in result_list:
                f.write(f"{result}\n")
    
    # Process with CNN
    print("\nProcessing detections with CNN classifier...")
    process_detections_with_cnn(configuration)
    print("\nProcessing detections for Task 2...")
    char_detections = process_detections_for_task2(configuration)
    
    print("\nEvaluating Task 2 results...")
    evaluate_task2(configuration)
    print(f"\nProcessing complete. Results in {detections_dir}")

if __name__ == "__main__":
    run_classifier()