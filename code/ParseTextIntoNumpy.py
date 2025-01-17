import os
import numpy as np

def create_task1_files(base_dir):
    """Create numpy files for task1 (general face detection)"""
    # Create task1 directory
    task1_dir = os.path.join(base_dir, 'task1')
    os.makedirs(task1_dir, exist_ok=True)
    
    # Read detection file
    detections_file = os.path.join(base_dir, 'task1gt_rezultate_detectie.txt')
    
    if not os.path.exists(detections_file):
        print(f"Detection file not found: {detections_file}")
        return
    
    # Parse detections
    detections = []  # [xmin, ymin, xmax, ymax]
    file_names = []  # image filenames
    
    with open(detections_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_names.append(parts[0])
                detections.append([int(x) for x in parts[1:5]])
    
    # Convert to numpy arrays
    detections_array = np.array(detections)
    file_names_array = np.array(file_names)
    scores_array = np.ones(len(detections))  # Default score of 1.0 for all detections
    
    # Save arrays
    np.save(os.path.join(task1_dir, 'detections_all_faces.npy'), detections_array)
    np.save(os.path.join(task1_dir, 'scores_all_faces.npy'), scores_array)
    np.save(os.path.join(task1_dir, 'file_names_all_faces.npy'), file_names_array)
    
    print(f"\nTask 1 files created in {task1_dir}")
    print(f"Total detections: {len(detections)}")

def create_task2_files(base_dir):
    """Create numpy files for task2 (character-specific detection)"""
    # Create task2 directory
    task2_dir = os.path.join(base_dir, 'task2')
    os.makedirs(task2_dir, exist_ok=True)
    
    # Character colors
    characters = ['dexter', 'deedee', 'mom', 'dad']
    
    # Initialize dictionaries for each character
    character_data = {char: {'detections': [], 'file_names': []} for char in characters}
    
    # Read CNN classifier results
    detections_dir = os.path.join(base_dir, 'detections')
    if not os.path.exists(detections_dir):
        print(f"Detections directory not found: {detections_dir}")
        return
    
    # Process final_* images that contain CNN classifications
    final_images = [f for f in os.listdir(detections_dir) if f.startswith('final_')]
    
    if not final_images:
        print("No processed images found. Please run CNN classifier first.")
        return
    
    # Read detection file for bounding boxes
    detections_file = os.path.join(base_dir, 'task1gt_rezultate_detectie.txt')
    detections_dict = {}
    
    with open(detections_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                img_name = parts[0]
                bbox = [int(x) for x in parts[1:5]]
                if img_name not in detections_dict:
                    detections_dict[img_name] = []
                detections_dict[img_name].append(bbox)
    
    # For each character, create numpy arrays
    for char in characters:
        detections = []
        file_names = []
        scores = []
        
        # Process each image's detections
        for img_name in detections_dict.keys():
            if img_name in detections_dict:
                for bbox in detections_dict[img_name]:
                    # In a real scenario, we would look up the CNN classification
                    # Here we're just using placeholder scores
                    detections.append(bbox)
                    file_names.append(img_name)
                    scores.append(1.0)  # Placeholder score
        
        # Save arrays for this character
        if detections:
            np.save(os.path.join(task2_dir, f'detections_{char}.npy'), np.array(detections))
            np.save(os.path.join(task2_dir, f'scores_{char}.npy'), np.array(scores))
            np.save(os.path.join(task2_dir, f'file_names_{char}.npy'), np.array(file_names))
            print(f"\n{char.capitalize()} files created:")
            print(f"Total detections: {len(detections)}")

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
