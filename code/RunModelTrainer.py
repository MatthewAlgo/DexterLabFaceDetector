from Parameters import *
from FacialDetectorDexter import *
import numpy as np
import os
import glob

def train_model():
    params = Parameters()
    facial_detector = FacialDetectorDexter(params)
    
    print("\nInformatii despre setul de date:")
    print(f"Numar maxim exemple pozitive per dimensiune: {params.number_positive_examples}")
    print(f"Numar maxim exemple negative per dimensiune: {params.number_negative_examples}")
    print(f"Dimensiuni ferestre: {params.sizes_array}")
    print(f"Numar total maxim exemple pozitive: {params.number_positive_examples * len(params.sizes_array)}")
    print(f"Numar total maxim exemple negative: {params.number_negative_examples * len(params.sizes_array)}")
    print(f"Total exemple de antrenare maxim: {(params.number_positive_examples + params.number_negative_examples) * len(params.sizes_array)}\n")
    
    # Check/generate descriptors
    print("Verificăm descriptorii...")
    all_files_exist = True
    descriptor_files = []
    
    for window_size in params.sizes_array:
        pos_file = os.path.join(params.dir_save_files, 
                               f'descriptoriExemplePozitive_{params.dim_hog_cell}_'
                               f'{params.number_positive_examples}_size_{window_size}.npy')
        neg_file = os.path.join(params.dir_save_files,
                               f'descriptoriExempleNegative_{params.dim_hog_cell}_'
                               f'{params.number_negative_examples}_size_{window_size}.npy')
        descriptor_files.append((window_size, pos_file, neg_file))
        if not (os.path.exists(pos_file) and os.path.exists(neg_file)):
            all_files_exist = False
            break
    
    # Load or generate descriptors
    positive_features_dict = {}
    negative_features_dict = {}
    
    if all_files_exist:
        print("Loading existing descriptors...")
        for window_size, pos_file, neg_file in descriptor_files:
            positive_features_dict[window_size] = np.load(pos_file)
            negative_features_dict[window_size] = np.load(neg_file)
            print(f'Loaded descriptors for window size {window_size}')
    else:
        print("Generating descriptors...")
        for window_size in params.sizes_array:
            print(f'Processing window size {window_size}...')
            positive_features_dict[window_size] = facial_detector.get_positive_descriptors(window_size)
            negative_features_dict[window_size] = facial_detector.get_negative_descriptors(window_size)
    
    # Prepare training data
    print("\nPreparing training data...")
    training_examples_dict = {}
    train_labels_dict = {}
    
    actual_pos = 0
    actual_neg = 0
    print("\nNumarul real de exemple procesate:")
    for window_size in params.sizes_array:
        pos_features = positive_features_dict[window_size]
        neg_features = negative_features_dict[window_size]
        actual_pos += len(pos_features)
        actual_neg += len(neg_features)
        print(f"Dimensiune {window_size}:")
        print(f"  Exemple pozitive: {len(pos_features)}")
        print(f"  Exemple negative: {len(neg_features)}")
        
        training_examples_dict[window_size] = np.concatenate((pos_features, neg_features))
        train_labels_dict[window_size] = np.concatenate((
            np.ones(len(pos_features)),
            np.zeros(len(neg_features))
        ))
        
        print(f"Window {window_size}: {len(training_examples_dict[window_size])} examples")
    
    print(f"\nTotal exemple pozitive procesate: {actual_pos}")
    print(f"Total exemple negative procesate: {actual_neg}")
    print(f"Total general: {actual_pos + actual_neg}")
    
    # Train classifier
    print("\nTraining classifier...")
    facial_detector.train_classifier(training_examples_dict, train_labels_dict)
    
    print("\nTraining completed successfully!")
    return facial_detector

if __name__ == "__main__":
    train_model()