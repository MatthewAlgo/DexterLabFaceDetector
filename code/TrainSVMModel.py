import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
from Parameters import Parameters
from skimage.feature import hog
import cv2 as cv

def compute_fixed_features(feature_vector, target_size=64):
    """Ensure all feature vectors have consistent dimensions"""
    if len(feature_vector.shape) == 1:
        # If already 1D, verify length is consistent
        expected_length = compute_expected_features_length(target_size)
        if len(feature_vector) == expected_length:
            return feature_vector
    
    # Compute expected feature length
    expected_length = compute_expected_features_length(target_size)
    
    # Return properly sized feature vector
    return np.resize(feature_vector, expected_length)

def compute_expected_features_length(window_size, cell_size=8):
    """Calculate expected HOG feature dimension for a given window size"""
    cells_per_block = 2
    block_stride = 1
    orientations = 9
    
    cells_in_window = window_size // cell_size
    blocks_in_window = cells_in_window - (cells_per_block - block_stride)
    
    features_per_block = cells_per_block * cells_per_block * orientations
    total_features = blocks_in_window * blocks_in_window * features_per_block
    
    return total_features

def train_unified_model(params: Parameters, training_examples, train_labels):
    """Enhanced training with better batching and learning rate scheduling"""
    print("\nTraining unified model...")
    
    model_file = os.path.join(params.dir_save_files, 'unified_svm_model.pkl')
    scaler_file = os.path.join(params.dir_save_files, 'unified_scaler.pkl')
    
    # Calculate class weights
    all_labels = []
    for window_size in params.sizes_array:
        all_labels.extend(train_labels[window_size])
    all_labels = np.array(all_labels)
    classes = np.array([0, 1])
    class_weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Enhanced SGD classifier with stronger parameters
    svm = SGDClassifier(
        loss='hinge',
        max_iter=10000,  # Increased iterations
        tol=1e-9,       # Tighter tolerance
        alpha=0.00001,   # Reduced regularization
        learning_rate='adaptive',
        eta0=0.1,       # Higher initial learning rate
        power_t=0.2,    # Slower learning rate decay
        class_weight=class_weight_dict,
        random_state=42,
        warm_start=True,
        average=10      # Use averaged SGD with more iterations
    )
    
    scaler = StandardScaler(with_mean=True, with_std=True)
    
    # Increased batch size for better stability
    batch_size = 4000  # Doubled from previous
    
    # Process and validate all features first
    print("\nPreprocessing and validating features...")
    valid_features_dict = {}
    valid_labels_dict = {}
    
    for window_size in params.sizes_array:
        pos_features = training_examples[window_size][:params.number_positive_examples]
        neg_features = training_examples[window_size][-params.number_negative_examples:]
        
        # Validate and collect features
        valid_pos = [f for f in pos_features if len(f) == params.expected_features]
        valid_neg = [f for f in neg_features if len(f) == params.expected_features]
        
        valid_features_dict[window_size] = np.vstack((valid_pos, valid_neg))
        valid_labels_dict[window_size] = np.concatenate((
            np.ones(len(valid_pos)),
            np.zeros(len(valid_neg))
        ))
        
        print(f"Window {window_size}: {len(valid_pos)} positive, {len(valid_neg)} negative")
    
    # Training loop with multiple epochs
    n_epochs = 10  # Increased from 3
    best_score = float('-inf')
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Shuffle window sizes for each epoch
        window_sizes = list(params.sizes_array)
        np.random.shuffle(window_sizes)
        
        for window_size in window_sizes:
            features = valid_features_dict[window_size]
            labels = valid_labels_dict[window_size]
            
            # Shuffle data
            shuffle_idx = np.random.permutation(len(features))
            features = features[shuffle_idx]
            labels = labels[shuffle_idx]
            
            # Process in batches
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                # Scale features
                if i == 0 and epoch == 0:
                    batch_features = scaler.fit_transform(batch_features)
                else:
                    batch_features = scaler.transform(batch_features)
                
                # Train on batch
                svm.partial_fit(batch_features, batch_labels, classes=[0, 1])
                
                # Compute and print loss
                score = svm.score(batch_features, batch_labels)
                if score > best_score:
                    best_score = score
                    # Save best model
                    with open(model_file, 'wb') as f:
                        pickle.dump(svm, f)
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(scaler, f)
                
                print(f"Batch score: {score:.4f} (best: {best_score:.4f})")
    
    print(f"\nTraining completed. Best score: {best_score:.4f}")
    return svm, scaler