from Parameters import *
from FacialDetectorDexter import *
from Visualize import *
import os
import multiprocessing as mp
from tqdm import tqdm
import glob
import torch

def apply_nms(boxes, scores, iou_threshold=0.3):
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return [], []
        
    # Convert to float
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep].astype(np.int32), scores[keep]

def process_image(args):
    """Process a single image with the detector"""
    image_path, model_file, scaler_file, params = args
    
    # Initialize detector for this process
    detector = FacialDetectorDexter(params)
    detector.load_classifier(model_file, scaler_file)
    
    # Process image
    image = cv.imread(image_path)
    image_name = os.path.basename(image_path)
    detections, scores = detector.detector.detect_faces(image)
    
    # Initialize CNN classifier
    cnn_classifier = CNNFaceClassifier.load_latest_model()
    
    # Apply NMS if there are detections
    if len(detections) > 0:
        detections, scores = apply_nms(detections, scores, iou_threshold=0.2)
        
        # Visualize and save only if detections remain after NMS
        if len(detections) > 0:
            viz_img = image.copy()
            results = []
            
            for det, score in zip(detections, scores):
                x1, y1, x2, y2 = map(int, det)
                
                # Get face classification
                face_img = image[y1:y2, x1:x2]
                face_img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
                member, conf, _ = cnn_classifier.predict(face_img_rgb)
                
                # Draw detection
                cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{member} ({conf:.2f})"
                cv.putText(viz_img, label, (x1, y1-5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Add to results with character classification
                results.append(f"{image_name} {x1} {y1} {x2} {y2} {member}")
            
            output_path = os.path.join(params.dir_save_files, 'detections', f'boxes_{image_name}')
            cv.imwrite(output_path, viz_img)
            
            return results
    return []

def process_detections_with_cnn(params):
    """Process detections in smaller batches"""
    # Read detection results
    results_file = os.path.join(params.dir_save_files, 'task1gt_rezultate_detectie.txt')
    with open(results_file, 'r') as f:
        detection_lines = f.readlines()
    
    # Initialize CNN classifier
    cnn_classifier = CNNFaceClassifier.load_latest_model()
    
    # Group detections by image
    detections_by_image = {}
    for line in detection_lines:
        parts = line.strip().split()
        image_name = parts[0]
        bbox = list(map(int, parts[1:]))
        if image_name not in detections_by_image:
            detections_by_image[image_name] = []
        detections_by_image[image_name].append(bbox)
    
    # Process each image's detections
    for image_name, bboxes in tqdm(detections_by_image.items(), desc="Processing images"):
        # Load image once
        image_path = os.path.join(params.dir_test_examples, image_name)
        image = cv.imread(image_path)
        if image is None:
            continue
            
        # Load visualization image
        viz_path = os.path.join(params.dir_save_files, 'detections', f'boxes_{image_name}')
        if not os.path.exists(viz_path):
            continue
        viz_img = cv.imread(viz_path)
        
        # Process all detections for this image
        results = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            face_img = image[y1:y2, x1:x2]
            face_img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
            
            # Get classification
            member, conf, _ = cnn_classifier.predict(face_img_rgb)
            results.append((bbox, member, conf))
        
        # Update visualization with all classifications
        for bbox, member, conf in results:
            x1, y1, x2, y2 = bbox
            # Add classification text
            label = f"{member} ({conf:.2f})"
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # Add text with background
            (label_w, label_h), _ = cv.getTextSize(label, font, font_scale, thickness)
            cv.rectangle(viz_img, (x1, y1-label_h-8), (x1+label_w+4, y1-4), (0, 0, 0), -1)
            cv.putText(viz_img, label, (x1+2, y1-8), font, font_scale, (255, 255, 255), thickness)
        
        # Save updated visualization
        output_path = os.path.join(params.dir_save_files, 'detections', f'final_{image_name}')
        cv.imwrite(output_path, viz_img)
        
        # Group results by character for task2
        char_results = {char: [] for char in ['dexter', 'deedee', 'mom', 'dad']}
        for bbox, member, conf in results:
            if member in char_results:
                char_results[member].append((bbox, conf, image_name))

def process_detections_for_task2(params):
    """Process detections and create separate files for each character"""
    results_file = os.path.join(params.dir_save_files, 'task1gt_rezultate_detectie.txt')
    
    # Dictionary to store character-specific detections
    char_detections = {
        'dexter': [], 'deedee': [], 'mom': [], 'dad': []
    }
    
    # Read all detections
    with open(results_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:  # We now have character in the line
                img_name, x1, y1, x2, y2, character = parts
                if character in char_detections:
                    char_detections[character].append(
                        f"{img_name} {x1} {y1} {x2} {y2}"
                    )
    
    # Save separate result files for each character
    for char in char_detections:
        output_file = os.path.join(params.dir_save_files, f'task2_{char}_rezultate.txt')
        with open(output_file, 'w') as f:
            for detection in char_detections[char]:
                f.write(f"{detection}\n")
    
    return char_detections

def evaluate_task2(params):
    """Evaluate Task 2 results against ground truth"""
    characters = ['dexter', 'deedee', 'mom', 'dad']
    
    for char in characters:
        print(f"\nEvaluating {char.capitalize()} detections:")
        
        # Read ground truth
        gt_file = os.path.join(params.dir_test_examples, '..', f'task2_{char}_gt_validare.txt')
        if not os.path.exists(gt_file):
            print(f"Ground truth file not found: {gt_file}")
            continue
            
        # Read results
        results_file = os.path.join(params.dir_save_files, f'task2_{char}_rezultate.txt')
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            continue
            
        # Load both files
        try:
            gt_data = np.loadtxt(gt_file, dtype=str)
            results_data = np.loadtxt(results_file, dtype=str)
            
            # Convert to proper format if needed
            if len(gt_data.shape) == 1:
                gt_data = gt_data.reshape(1, -1)
            if len(results_data.shape) == 1:
                results_data = results_data.reshape(1, -1)
                
            # Calculate metrics
            print(f"Ground truth detections: {len(gt_data)}")
            print(f"Algorithm detections: {len(results_data)}")
            
            # Detailed evaluation could be added here
            
        except Exception as e:
            print(f"Error evaluating {char}: {e}")

def run_classifier():
    params = Parameters()
    
    # Create detections directory
    detections_dir = os.path.join(params.dir_save_files, 'detections')
    os.makedirs(detections_dir, exist_ok=True)
    
    # Get model paths
    model_file = os.path.join(params.dir_save_files, 'unified_svm_model.pkl')
    scaler_file = os.path.join(params.dir_save_files, 'unified_scaler.pkl')
    
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        print("Please train the model first")
        return
    
    # Get all test images
    test_images = glob.glob(os.path.join(params.dir_test_examples, '*.jpg'))
    
    # Prepare arguments for parallel processing
    process_args = [(img, model_file, scaler_file, params) for img in test_images]
    
    # Create output file
    output_file = os.path.join(params.dir_save_files, 'task1gt_rezultate_detectie.txt')
    
    # Increase number of processes
    num_processes = min(12, mp.cpu_count())  # Increased from 8 to 12
    print(f"\nProcessing images with {num_processes} processes...")
    
    # Run parallel processing with progress bar
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, process_args), 
                          total=len(test_images),
                          desc="Processing images"))
    
    # Write results to file
    with open(output_file, 'w') as f:
        for result_list in results:
            for result in result_list:
                f.write(f"{result}\n")
    
    # Second pass: Process all detections with CNN
    print("\nProcessing detections with CNN classifier...")
    process_detections_with_cnn(params)
    
    # Process detections for Task 2
    print("\nProcessing detections for Task 2...")
    char_detections = process_detections_for_task2(params)
    
    # Evaluate Task 2 results
    print("\nEvaluating Task 2 results...")
    evaluate_task2(params)
    
    print(f"\nProcessing complete. Results in {detections_dir}")

if __name__ == "__main__":
    run_classifier()
