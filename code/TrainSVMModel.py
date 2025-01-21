import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
from Parameters import Parameters
from skimage.feature import hog
import cv2 as cv

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
    """Enhanced training with better feature validation"""
    print("\nTraining unified model...")
    
    model_file = os.path.join(params.dir_save_files, 'unified_svm_model.pkl')
    scaler_file = os.path.join(params.dir_save_files, 'unified_scaler.pkl')
    
    # Better feature validation
    print("\nPreprocessing and validating features...")
    valid_features = []
    valid_labels = []
    
    expected_dim = params.expected_features
    print(f"Expected feature dimension: {expected_dim}")
    
    for window_size in params.sizes_array:
        features = training_examples[window_size]
        labels = train_labels[window_size]
        
        # Verify feature dimensions
        valid_indices = []
        for i, feat in enumerate(features):
            if feat is not None and len(feat) == expected_dim:
                valid_indices.append(i)
                
        if valid_indices:
            valid_features.extend(features[valid_indices])
            valid_labels.extend(labels[valid_indices])
            
        print(f"Window {window_size}: {len(valid_indices)} valid examples of {len(features)} total")
    
    if len(valid_features) == 0:
        raise ValueError(f"No valid features found! Expected dimension: {expected_dim}")
    
    print(f"\nTotal valid examples: {len(valid_features)}")
    
    # Convert to numpy arrays
    valid_features = np.array(valid_features)
    valid_labels = np.array(valid_labels)
    
    print(f"\nTotal valid examples: {len(valid_features)}")
    print(f"Positive examples: {np.sum(valid_labels == 1)}")
    print(f"Negative examples: {np.sum(valid_labels == 0)}")
    
    # Pre-compute class weights
    pos_count = np.sum(valid_labels == 1)
    neg_count = np.sum(valid_labels == 0)
    total = len(valid_labels)
    
    # Calculate balanced weights
    class_weight_dict = {
        0: total / (2.0 * neg_count),
        1: total / (2.0 * pos_count)
    }
    
    # Initialize scaler
    scaler = StandardScaler()
    valid_features = scaler.fit_transform(valid_features)
    
    # Initialize classifier without 'balanced' class_weight
    svm = SGDClassifier(
        loss='hinge',
        max_iter=20000,        # More iterations
        tol=1e-10,            # Tighter tolerance
        alpha=0.000001,       # Less regularization
        learning_rate='optimal',
        eta0=0.1,
        power_t=0.1,          # Slower learning rate decay
        class_weight=class_weight_dict,  # Using pre-computed weights
        random_state=42,
        warm_start=True,
        average=True          # Use averaged SGD
    )
    
    # Better training loop
    batch_size = 2000        # Smaller batches
    n_epochs = 15            # More epochs
    best_score = float('-inf')
    patience = 3             # Early stopping patience
    no_improve = 0
    
    print("\nStarting training...")
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Shuffle data
        shuffle_idx = np.random.permutation(len(valid_features))
        features_shuffled = valid_features[shuffle_idx]
        labels_shuffled = valid_labels[shuffle_idx]
        
        # Process in batches
        for i in range(0, len(features_shuffled), batch_size):
            batch_features = features_shuffled[i:i + batch_size]
            batch_labels = labels_shuffled[i:i + batch_size]
            
            # Train on batch
            svm.partial_fit(batch_features, batch_labels, classes=[0, 1])
            
            # Compute score
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