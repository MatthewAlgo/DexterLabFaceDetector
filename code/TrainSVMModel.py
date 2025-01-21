import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
from Parameters import Parameters
from skimage.feature import hog
import cv2 as cv

def calculate_hog_feature_dimension(window_size, cell_size=8):
    num_cells_per_block = 2
    block_shift = 1
    num_orientation_bins = 9
    cells_count_in_window = window_size // cell_size
    num_blocks_in_window = cells_count_in_window - (num_cells_per_block - block_shift)
    
    features_in_block = num_cells_per_block * num_cells_per_block * num_orientation_bins
    block_feature_count = num_blocks_in_window * num_blocks_in_window * features_in_block
    
    return block_feature_count

def train_unified_model(params: Parameters, training_examples, train_labels):
    # Train a unified model with all window sizes
    print("\nTraining unified model...")
    model_file = os.path.join(params.dir_save_files, 'unified_svm_model.pkl')
    scaler_file = os.path.join(params.dir_save_files, 'unified_scaler.pkl')
    
    print("\nPreprocessing and validating features...")
    processed_features = []
    filtered_labels = []
    
    expected_feature_dimension = params.computed_feature_count
    print(f"Expected feature dimension: {expected_feature_dimension}")
    for window_size in params.sizes_array:
        features = training_examples[window_size]
        labels = train_labels[window_size]
        
        valid_feature_indices = []
        for i, feat in enumerate(features):
            if feat is not None and len(feat) == expected_feature_dimension:
                valid_feature_indices.append(i)
                
        if valid_feature_indices:
            processed_features.extend(features[valid_feature_indices])
            filtered_labels.extend(labels[valid_feature_indices])
            
        print(f"Window {window_size}: {len(valid_feature_indices)} valid examples of {len(features)} total")
    
    if len(processed_features) == 0:
        raise ValueError(f"No valid features found! Expected dimension: {expected_feature_dimension}")
    
    print(f"\nTotal valid examples: {len(processed_features)}")
    
    # Convert to numpy
    processed_features = np.array(processed_features)
    filtered_labels = np.array(filtered_labels)
    print(f"\nTotal valid examples: {len(processed_features)}")
    print(f"Positive examples: {np.sum(filtered_labels == 1)}")
    print(f"Negative examples: {np.sum(filtered_labels == 0)}")
    positive_example_count = np.sum(filtered_labels == 1)
    negative_example_count = np.sum(filtered_labels == 0)
    total_example_count = len(filtered_labels)
    
    # Balance class weights
    class_balance_weights = { 0: total_example_count / (2.0 * negative_example_count), 1: total_example_count / (2.0 * positive_example_count)}
    
    # Scale features for normalization
    feature_normalizer = StandardScaler()
    processed_features = feature_normalizer.fit_transform(processed_features)
    
    svm_model_instance = SGDClassifier( loss='hinge', max_iter=20000, tol=1e-10, alpha=0.000001, learning_rate='optimal',
        eta0=0.1, power_t=0.1, class_weight=class_balance_weights, random_state=42, warm_start=True, average=True)
    
    training_batch_size = 2000        # Smaller batches
    num_training_epochs = 15            # More epochs
    highest_training_score = float('-inf')
    
    print("\nStarting training...")
    for epoch in range(num_training_epochs):
        print(f"\nEpoch {epoch + 1}/{num_training_epochs}")
        
        # Shuffle data
        shuffled_indices = np.random.permutation(len(processed_features))
        shuffled_training_features = processed_features[shuffled_indices]
        shuffled_training_labels = filtered_labels[shuffled_indices]
        for i in range(0, len(shuffled_training_features), training_batch_size):
            batch_features = shuffled_training_features[i:i + training_batch_size]
            batch_labels = shuffled_training_labels[i:i + training_batch_size]
            
            # Train on batch
            svm_model_instance.partial_fit(batch_features, batch_labels, classes=[0, 1])
            # Compute score
            score = svm_model_instance.score(batch_features, batch_labels)
            if score > highest_training_score:
                highest_training_score = score
                # Save best
                with open(model_file, 'wb') as f:
                    pickle.dump(svm_model_instance, f)
                with open(scaler_file, 'wb') as f:
                    pickle.dump(feature_normalizer, f)
            
            print(f"Batch score: {score:.4f} (best: {highest_training_score:.4f})")
    
    print(f"\nTraining completed. Best score: {highest_training_score:.4f}")
    return svm_model_instance, feature_normalizer