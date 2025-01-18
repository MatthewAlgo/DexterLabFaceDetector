from Parameters import *
from FacialDetectorDexter import *
from Visualize import *
import os
import multiprocessing as mp
from tqdm import tqdm
import glob
import torch
import time  # Add this import

def apply_nms(boxes, scores, iou_threshold=0.3):
    """Improved NMS with better overlap handling"""
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
    
    # Calculate areas once
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # Calculate IoU efficiently
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # Calculate IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Get indices of boxes with IoU <= threshold
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep].astype(np.int32), scores[keep]

def process_batch(args):
    """Process a batch of images in parallel"""
    image_paths, model_file, scaler_file, params = args
    results = []
    
    # Load model and scaler once per process
    detector = FacialDetectorDexter(params)
    detector.load_classifier(model_file, scaler_file)
    
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image = cv.imread(image_path)
        if image is None:
            continue
        
        # Run detection
        detections, scores = detector.detector.detect_faces_parallel(image, params.window_size, detector.model, detector.scaler)
        
        if len(detections) > 0:
            # Apply NMS
            detections, scores = apply_nms(detections, scores, iou_threshold=0.2)
            
            # Save results
            output_path = os.path.join(params.dir_save_files, 'detections', f'boxes_{image_name}')
            viz_img = image.copy()
            
            # Save detection results
            for det, score in zip(detections, scores):
                x1, y1, x2, y2 = map(int, det)
                results.append(f"{image_name} {x1} {y1} {x2} {y2}")
                cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(viz_img, f"{score:.2f}", (x1, y1-5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv.imwrite(output_path, viz_img)
    
    return results

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

def process_image_batch(args):
    """Process a single image in the parallel batch"""
    image_path, params, detector = args
    
    image_name = os.path.basename(image_path)
    start_time = time.time()
    
    print(f"\nProcesez imaginea: {image_name}")
    
    # Load and process image
    image = cv.imread(image_path)
    if image is None:
        print(f"Eroare la încărcarea imaginii: {image_name}")
        return None
        
    # Process each window size
    all_detections = []
    all_scores = []
    
    for window_size in params.sizes_array:
        if window_size not in detector.models:
            continue
            
        print(f"  -> Procesez fereastra de dimensiune {window_size}x{window_size}")
        detections, scores = detector.detect_faces_at_scale(
            image, 
            window_size,
            detector.models[window_size],
            detector.scalers[window_size]
        )
        
        if len(detections) > 0:
            all_detections.extend(detections)
            all_scores.extend(scores)
    
    # Process results
    output_path = None
    final_dets = []
    final_scores = []
    
    if len(all_detections) > 0:
        all_detections = np.array(all_detections)
        all_scores = np.array(all_scores)
        
        # Apply NMS
        final_dets, final_scores = apply_nms(all_detections, all_scores)
        
        # Create visualization
        viz_img = image.copy()
        for det, score in zip(final_dets, final_scores):
            x1, y1, x2, y2 = map(int, det)
            cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(viz_img, f"{score:.2f}", (x1, y1-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save detection visualization
        output_path = os.path.join(params.dir_save_files, 'detections', f'detection_{image_name}')
        cv.imwrite(output_path, viz_img)
    
    end_time = time.time()
    print(f"✓ Finalizat procesarea pentru {image_name} în {end_time - start_time:.2f} secunde")
    print(f"  -> {len(final_dets)} detecții găsite")
    
    return {
        'image_name': image_name,
        'detections': final_dets,
        'scores': final_scores,
        'output_path': output_path,
        'processing_time': end_time - start_time
    }

def process_detection_batch(args):
    """Optimized batch processing"""
    image_path, params, models, scalers = args
    image_name = os.path.basename(image_path)
    
    # Load image once
    image = cv.imread(image_path)
    if image is None:
        return None
        
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    
    all_detections = []
    all_scores = []
    
    for window_size in params.sizes_array:
        if window_size not in models:
            continue
            
        model = models[window_size]
        scaler = scalers[window_size]
        
        # Update window size in params
        params.window_size = window_size
        
        # Compute stride based on window size
        stride = max(8, window_size // 16)
        
        # Slide window
        for y in range(0, gray.shape[0] - window_size + 1, stride):
            for x in range(0, gray.shape[1] - window_size + 1, stride):
                window = gray[y:y + window_size, x:x + window_size].copy()
                features = TrainSVMModel.compute_hog_features(window, params)
                
                if features is not None:
                    features_scaled = scaler.transform([features])
                    score = model.decision_function(features_scaled)[0]
                    
                    if score >= params.threshold:
                        all_detections.append([x, y, x + window_size, y + window_size])
                        all_scores.append(score)
    
    # Apply NMS if detections exist
    if len(all_detections) > 0:
        final_dets, final_scores = apply_nms(np.array(all_detections), np.array(all_scores))
        
        # Save visualization
        viz_img = image.copy()
        for det, score in zip(final_dets, final_scores):
            x1, y1, x2, y2 = map(int, det)
            cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(viz_img, f"{score:.2f}", (x1, y1-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        output_path = os.path.join(params.dir_save_files, 'detections', f'detection_{image_name}')
        cv.imwrite(output_path, viz_img)
        
        return {
            'image_name': image_name,
            'detections': final_dets,
            'scores': final_scores
        }
    
    return None

def process_single_image(args):
    """Process a single image with both standard and varied windows"""
    image_path, params, detector = args
    image_name = os.path.basename(image_path)
    start_time = time.time()
    
    # Load and preprocess image once
    image = cv.imread(image_path)
    if image is None:
        return None
        
    all_detections = []
    all_scores = []
    
    # 1. Process standard window sizes
    for window_size in params.sizes_array:
        if window_size not in detector.models:
            continue
            
        dets, scores = detector.detect_faces_at_scale(
            image, 
            window_size,
            detector.models[window_size],
            detector.scalers[window_size]
        )
        if len(dets) > 0:
            all_detections.extend(dets)
            all_scores.extend(scores)
    
    # 2. Process varied windows with 140x140 model if available
    if 140 in detector.models:
        varied_dets, varied_scores = detector.detect_with_varied_windows(
            image,
            detector.models[140],
            detector.scalers[140]
        )
        if len(varied_dets) > 0:
            all_detections.extend(varied_dets)
            all_scores.extend(varied_scores)
    
    # 3. Apply global NMS
    if len(all_detections) > 0:
        all_detections = np.array(all_detections)
        all_scores = np.array(all_scores)
        
        final_dets, final_scores = apply_nms(
            all_detections,
            all_scores,
            iou_threshold=0.2  # More aggressive threshold for final pass
        )
        
        # Create visualization
        viz_img = image.copy()
        for det, score in zip(final_dets, final_scores):
            x1, y1, x2, y2 = map(int, det)
            cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(viz_img, f"{score:.2f}", (x1, y1-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        output_path = os.path.join(params.dir_save_files, 'detections', f'detection_{image_name}')
        cv.imwrite(output_path, viz_img)
        
        end_time = time.time()
        return {
            'image_name': image_name,
            'detections': final_dets,
            'scores': final_scores,
            'processing_time': end_time - start_time
        }
    
    return None

def generate_varied_windows(min_size=60, max_size=220, num_sizes=10):
    """Generate window sizes with varied aspect ratios"""
    widths = np.linspace(min_size, max_size, num_sizes, dtype=int)
    heights = np.linspace(min_size, max_size, num_sizes, dtype=int)
    
    windows = []
    for w, h in itertools.product(widths, heights):
        aspect_ratio = w / h
        if 0.5 <= aspect_ratio <= 2.0:
            windows.append((w, h))
    
    if len(windows) > 100:
        indices = np.linspace(0, len(windows)-1, 100, dtype=int)
        windows = [windows[i] for i in indices]
    
    return windows

def detect_faces_at_varied_scale(image, window_size, model, scaler, params):
    """Detect faces using a varied window size but fixed model"""
    w, h = window_size
    
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv.equalizeHist(gray)
    
    detections = []
    scores = []
    
    # Adaptive stride
    stride_w = max(8, w // 16)
    stride_h = max(8, h // 16)
    
    for y in range(0, gray.shape[0] - h + 1, stride_h):
        for x in range(0, gray.shape[1] - w + 1, stride_w):
            window = gray[y:y+h, x:x+w]
            # Resize to 140x140 for feature extraction
            window_resized = cv.resize(window, (140, 140))
            
            features = TrainSVMModel.compute_hog_features(window_resized, params)
            
            if features is not None:
                features_scaled = scaler.transform([features])
                score = model.decision_function(features_scaled)[0]
                
                if score >= params.threshold:
                    detections.append([x, y, x + w, y + h])
                    scores.append(score)
    
    return np.array(detections), np.array(scores)

def run_detector(params):
    """Run parallel face detection"""
    detector = FacialDetectorDexter(params)
    
    if not detector.models or not detector.scalers:
        print("No models loaded. Please train the model first.")
        return
    
    # Create output directories
    detections_dir = os.path.join(params.dir_save_files, 'detections')
    os.makedirs(detections_dir, exist_ok=True)
    
    # Get test images
    test_images = glob.glob(os.path.join(params.dir_test_examples, '*.jpg'))
    total_images = len(test_images)
    
    print(f"\nStarting detection for {total_images} images using {min(12, total_images)} processes...")
    
    # Initialize multiprocessing with spawn
    mp.set_start_method('spawn', force=True)
    
    # Process images in parallel
    with mp.Pool(processes=min(12, total_images)) as pool:
        process_args = [(img_path, params, detector) 
                       for img_path in test_images]
        
        results = list(tqdm(
            pool.imap(process_single_image, process_args),
            total=len(test_images),
            desc="Processing images"
        ))
    
    # Save detections to file
    save_detections_to_file(results, params)
    
    # Print statistics
    successful = [r for r in results if r is not None]
    total_dets = sum(len(r['detections']) if r and 'detections' in r else 0 for r in successful)
    total_time = sum(r['processing_time'] for r in successful)
    avg_time = total_time / len(successful) if successful else 0
    
    print(f"\nProcessing complete:")
    print(f"Images processed: {len(successful)}/{total_images}")
    print(f"Total detections: {total_dets}")
    print(f"Average time per image: {avg_time:.2f} seconds")
    print(f"\nResults saved to:")
    print(f"- Detections: {detections_dir}")
    print(f"- Results file: {os.path.join(params.dir_save_files, 'task1gt_rezultate_detectie.txt')}")

def save_detections_to_file(results, params):
    """Save detection results to text file"""
    output_file = os.path.join(params.dir_save_files, 'task1gt_rezultate_detectie.txt')
    with open(output_file, 'w') as f:
        for result in results:
            if result and 'detections' in result and len(result['detections']) > 0:
                image_name = result['image_name']
                for det in result['detections']:
                    # Format: image_name x1 y1 x2 y2
                    detection_str = f"{image_name} {' '.join(map(str, map(int, det)))}\n"
                    f.write(detection_str)

def run_cnn_classifier(params):
    """Run CNN classification on detected faces"""
    # Initialize CNN classifier
    cnn_classifier = CNNFaceClassifier.load_latest_model()
    
    # Read detection results
    detections_file = os.path.join(params.dir_save_files, 'task1gt_rezultate_detectie.txt')
    if not os.path.exists(detections_file):
        print(f"No detections file found at {detections_file}")
        return
        
    # Process detections
    process_detections_with_cnn(params, cnn_classifier)

def process_detections_with_cnn(params, cnn_classifier):
    """Process existing detections with CNN classifier"""
    results_file = os.path.join(params.dir_save_files, 'task1gt_rezultate_detectie.txt')
    
    # Read and group detections by image
    detections_by_image = {}
    with open(results_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            bbox = list(map(int, parts[1:5]))
            if img_name not in detections_by_image:
                detections_by_image[img_name] = []
            detections_by_image[img_name].append(bbox)
    
    print(f"\nClassifying faces in {len(detections_by_image)} images...")
    
    # Process each image
    for image_name in tqdm(detections_by_image.keys(), desc="Classifying faces"):
        process_single_image_cnn(image_name, detections_by_image[image_name], 
                               params, cnn_classifier)

def run_classifier():
    params = Parameters()
    detector = FacialDetectorDexter(params)
    
    if not detector.models or not detector.scalers:
        print("No models loaded. Please train the model first.")
        return
        
    # Create output directories
    detections_dir = os.path.join(params.dir_save_files, 'detections')
    os.makedirs(detections_dir, exist_ok=True)
    
    # Get all test images
    test_images = glob.glob(os.path.join(params.dir_test_examples, '*.jpg'))
    total_images = len(test_images)
    
    print(f"\nÎncep procesarea pentru {total_images} imagini în paralel...")
    
    # Split images into batches
    num_processes = 12
    batch_size = len(test_images) // num_processes + (1 if len(test_images) % num_processes else 0)
    batches = [test_images[i:i + batch_size] for i in range(0, len(test_images), batch_size)]
    
    # Process batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        args = [(img_path, params, detector) for img_path in test_images]
        results = list(tqdm(pool.imap(process_image_batch, args), 
                          total=len(test_images),
                          desc="Procesare imagini"))
    
    # Process results
    successful_detections = [r for r in results if r is not None]
    
    print("\nStatistici procesare:")
    print(f"Total imagini procesate: {len(successful_detections)}/{total_images}")
    total_detections = sum(len(r['detections']) for r in successful_detections)
    total_time = sum(r['processing_time'] for r in successful_detections)
    print(f"Total detecții: {total_detections}")
    print(f"Timp total procesare: {total_time:.2f} secunde")
    print(f"Timp mediu per imagine: {total_time/len(successful_detections):.2f} secunde")
    print("\nProcesare completă! Toate rezultatele au fost salvate în directorul 'detections'")

if __name__ == "__main__":
    run_classifier()
