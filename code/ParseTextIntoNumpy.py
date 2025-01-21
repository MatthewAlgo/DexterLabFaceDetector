import os
import numpy as np
from tqdm import tqdm
import cv2 as cv

def create_task1_files(base_dir):
    """Create numpy files for task1 (general face detection)"""
    print("\n" + "="*80)
    print("TASK 1: Processing General Face Detection Results")
    print("="*80)
    
    # Create directory for task1
    task1_dir = os.path.join(base_dir, 'task1')
    os.makedirs(task1_dir, exist_ok=True)
    print(f"\nCreated/verified task1 directory: {task1_dir}")
    
    face_detection_results_file = os.path.join(base_dir, 'task1gt_rezultate_detectie.txt')
    if not os.path.exists(face_detection_results_file):
        print(f"Detection file not found: {face_detection_results_file}")
        return
    
    print("\nParsing detections file...")
    print("Looking for valid detections with format: <filename> <x1> <y1> <x2> <y2>")
    
    face_detections = []
    detected_face_file_names = []
    invalid_coordinate_count = 0
    empty_line_count = 0
    
    with open(face_detection_results_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="Processing detection lines")):
            line = line.strip()
            if not line:
                empty_line_count += 1
                continue
                
            parts = line.split()
            if len(parts) >= 5:
                try:
                    coords = [int(x) for x in parts[1:5]]
                    if all(c >= 0 for c in coords) and coords[2] > coords[0] and coords[3] > coords[1]:
                        detected_face_file_names.append(parts[0])
                        face_detections.append(coords)
                        if (len(face_detections) % 100) == 0:
                            print(f"\nProcessed {len(face_detections)} valid detections so far...")
                    else:
                        print(f"\nWarning: Invalid coordinates in line {i+1}: {line}")
                        invalid_coordinate_count += 1
                except ValueError:
                    print(f"\nError: Non-numeric coordinates in line {i+1}: {line}")
                    invalid_coordinate_count += 1
            else:
                print(f"\nWarning: Incorrect format in line {i+1}: {line}")
                invalid_coordinate_count += 1
    
    print("\nDetection Processing Summary:")
    print(f"Total lines processed: {len(lines)}")
    print(f"Valid detections found: {len(face_detections)}")
    print(f"Invalid lines: {invalid_coordinate_count}")
    print(f"Empty/skipped lines: {empty_line_count}")
    
    if not face_detections:
        print("\nWARNING: No valid detections found!")
        print("Creating placeholder data to avoid empty files...")
        face_detections = [[0, 0, 100, 100]]
        detected_face_file_names = ["placeholder.jpg"]
    
    print("\nConverting to numpy arrays...")
    detections_array = np.array(face_detections, dtype=np.int32)
    file_names_array = np.array(detected_face_file_names, dtype=str)
    scores_array = np.ones(len(face_detections), dtype=np.float32)
    
    print("\nSaving numpy arrays...")
    print(f"Detections shape: {detections_array.shape}")
    print(f"File names shape: {file_names_array.shape}")
    print(f"Scores shape: {scores_array.shape}")
    
    # Save arrays
    for name, array in [
        ('detections_all_faces.npy', detections_array),
        ('scores_all_faces.npy', scores_array),
        ('file_names_all_faces.npy', file_names_array)
    ]:
        path = os.path.join(task1_dir, name)
        np.save(path, array)
        print(f"Saved: {path}")
    
    print("\nTask 1 Processing Complete!")
    print("="*80)

def create_task2_files(base_dir):
    """Create numpy files for task2 (character-specific detection)"""
    print("\n" + "="*80)
    print("TASK 2: Processing Character-Specific Detections")
    print("="*80)
    
    task2_dir = os.path.join(base_dir, 'task2')
    os.makedirs(task2_dir, exist_ok=True)
    print(f"\nCreated/verified task2 directory: {task2_dir}")
    
    # Read task1 results
    task1_detections_file = os.path.join(base_dir, 'task1gt_rezultate_detectie.txt')
    if not os.path.exists(task1_detections_file):
        print("Task 1 detection file not found. Please run task 1 first.")
        return
    
    print("\nLoading CNN classifier...")
    try:
        from RunCNNFaceClassifier import CNNFaceClassifier
        cnn_classifier = CNNFaceClassifier.load_latest_model()
        print("CNN classifier loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load CNN classifier: {e}")
        return
    
    # Initialize character data
    character_list = ['dexter', 'deedee', 'mom', 'dad']
    character_detection_data = {char: {'detections': [], 'file_names': [], 'scores': []} for char in character_list}
    
    print("\nReading Task 1 detections...")
    character_bounding_boxes = []
    detected_image_names = []
    with open(task1_detections_file, 'r') as f:
        lines = f.readlines()
        print(f"Found {len(lines)} detection lines to process")
        
        for line in tqdm(lines, desc="Reading detections"):
            parts = line.strip().split()
            if len(parts) >= 5:
                image_filename = parts[0]
                face_bounding_box = list(map(int, parts[1:5]))
                character_bounding_boxes.append(face_bounding_box)
                detected_image_names.append(image_filename)
    
    print(f"\nProcessing {len(character_bounding_boxes)} detections with CNN classifier...")
    
    images_base_dir = os.path.join(base_dir, '..', '..', 'validare', 'validare')
    print(f"Image directory: {images_base_dir}")
    
    detected_character_counts = {char: 0 for char in character_list}
    errors = 0
    
    for i, (face_bounding_box, image_filename) in enumerate(tqdm(zip(character_bounding_boxes, detected_image_names), total=len(character_bounding_boxes), desc="Classifying faces")):
        image_file_path = os.path.join(images_base_dir, image_filename)
        
        try:
            image = cv.imread(image_file_path)
            if image is None:
                print(f"\nWarning: Could not load image: {image_file_path}")
                continue
            
            bounding_box_x1, bounding_box_top, bounding_box_x_end, bounding_box_bottom = face_bounding_box
            face_img = image[bounding_box_top:bounding_box_bottom, bounding_box_x1:bounding_box_x_end]
            face_img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
            
            member, confidence, probabilities = cnn_classifier.predict(face_img_rgb)
            
            if member in character_list and confidence > 0.5:
                character_detection_data[member]['detections'].append(face_bounding_box)
                character_detection_data[member]['file_names'].append(image_filename)
                character_detection_data[member]['scores'].append(confidence)
                detected_character_counts[member] += 1
                
                if confidence > 0.9:
                    print(f"\nHigh confidence detection: {member} ({confidence:.3f}) in {image_filename}")
            
        except Exception as e:
            print(f"\nError processing {image_filename}: {e}")
            errors += 1
    
    print("\nClassification Summary:")
    print("-"*40)
    for char in character_list:
        print(f"{char.capitalize()}: {detected_character_counts[char]} detections")
    print(f"Errors encountered: {errors}")
    
    print("\nSaving character-specific results...")
    for char in character_list:
        if not character_detection_data[char]['detections']:
            print(f"\nNo detections for {char}, adding placeholder...")
            character_detection_data[char]['detections'].append([0, 0, 100, 100])
            character_detection_data[char]['file_names'].append("placeholder.jpg")
            character_detection_data[char]['scores'].append(0.1)
        
        # Convert and save arrays
        character_bounding_boxes = np.array(character_detection_data[char]['detections'], dtype=np.int32)
        detected_image_names = np.array(character_detection_data[char]['file_names'], dtype=str)
        scores = np.array(character_detection_data[char]['scores'], dtype=np.float32)
        
        print(f"\nSaving {char.capitalize()} data:")
        print(f"Detections shape: {character_bounding_boxes.shape}")
        print(f"Scores range: {scores.min():.3f} - {scores.max():.3f}")
        
        # Save arrays
        for name, array in [
            (f'detections_{char}.npy', character_bounding_boxes),
            (f'scores_{char}.npy', scores),
            (f'file_names_{char}.npy', detected_image_names)
        ]:
            path = os.path.join(task2_dir, name)
            np.save(path, array)
            print(f"Saved: {path}")
    
    print("\nTask 2 Processing Complete!")
    print("="*80)

def main():
    base_dir = os.path.join('..', 'antrenare', 'fisiere_salvate_algoritm')
    print("Choose task to process:")
    print("1. Create Task 1 files (general face detection)")
    print("2. Create Task 2 files (character recognition)")
    print("3. Both")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        create_task1_files(base_dir)
    elif choice == "2":
        create_task2_files(base_dir)
    elif choice == "3":
        create_task1_files(base_dir)
        create_task2_files(base_dir)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()