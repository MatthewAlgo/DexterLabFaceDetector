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
    # Gen descriptors
    print("Verificam descriptorii...")
    descriptor_files_exist = True
    descriptor_files = []
    
    for window_size in params.sizes_array:
        positive_descriptor_file = os.path.join(params.dir_save_files, 
                               f'descriptoriExemplePozitive_{params.hog_cell_dimension}_'
                               f'{params.number_positive_examples}_size_{window_size}.npy')
        negative_descriptor_file = os.path.join(params.dir_save_files,
                               f'descriptoriExempleNegative_{params.hog_cell_dimension}_'
                               f'{params.number_negative_examples}_size_{window_size}.npy')
        descriptor_files.append((window_size, positive_descriptor_file, negative_descriptor_file))
        if not (os.path.exists(positive_descriptor_file) and os.path.exists(negative_descriptor_file)):
            descriptor_files_exist = False
            break
    
    # Load or generate descriptors
    positive_features_dict = {}
    negative_features_dict = {}
    
    if descriptor_files_exist:
        print("Loading existing descriptors...")
        for window_size, positive_descriptor_file, negative_descriptor_file in descriptor_files:
            positive_features_dict[window_size] = np.load(positive_descriptor_file)
            negative_features_dict[window_size] = np.load(negative_descriptor_file)
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
    
    actual_positive_count = 0
    actual_negative_count = 0
    print("\nNumarul real de exemple procesate:")
    for window_size in params.sizes_array:
        pos_features = positive_features_dict[window_size]
        neg_features = negative_features_dict[window_size]
        actual_positive_count += len(pos_features)
        actual_negative_count += len(neg_features)
        print(f"Dimensiune {window_size}:")
        print(f"  Exemple pozitive: {len(pos_features)}")
        print(f"  Exemple negative: {len(neg_features)}")
        
        training_examples_dict[window_size] = np.concatenate((pos_features, neg_features))
        train_labels_dict[window_size] = np.concatenate((
            np.ones(len(pos_features)),
            np.zeros(len(neg_features))
        ))
        
        print(f"Window {window_size}: {len(training_examples_dict[window_size])} examples")
    
    print(f"\nTotal exemple pozitive procesate: {actual_positive_count}")
    print(f"Total exemple negative procesate: {actual_negative_count}")
    print(f"Total general: {actual_positive_count + actual_negative_count}")
    print("\nTraining classifier...")
    facial_detector.train_classifier(training_examples_dict, train_labels_dict)
    
    print("\nTraining completed successfully!")
    return facial_detector

if __name__ == "__main__":
    train_model()