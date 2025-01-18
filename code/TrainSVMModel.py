import numpy as np
from sklearn.svm import SVC  # Changed from LinearSVC to SVC for RBF kernel
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
from Parameters import Parameters
from skimage.feature import hog
import cv2 as cv
from functools import partial
import multiprocessing as mp
from itertools import product
from tqdm import tqdm

# Make this function available for import
__all__ = ['compute_hog_features', 'train_unified_model']

def compute_hog_features(image, params):
    """Enhanced HOG feature computation with corrected dimension checking"""
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Basic preprocessing
    gray = cv.equalizeHist(gray)
    gray = cv.resize(gray, (80, 80))  # Fixed size for consistent features
    
    try:
        hog_params = params.hog_params.copy()
        features = hog(
            gray,
            orientations=hog_params['orientations'],
            pixels_per_cell=hog_params['pixels_per_cell'],
            cells_per_block=hog_params['cells_per_block'],
            block_norm=hog_params['block_norm'],
            transform_sqrt=hog_params['transform_sqrt'],
            feature_vector=True
        )
        
        if len(features) == params.expected_features:
            return features
            
        # More informative error message
        cells_y = gray.shape[0] // hog_params['pixels_per_cell'][0]
        cells_x = gray.shape[1] // hog_params['pixels_per_cell'][1]
        blocks_y = cells_y - hog_params['cells_per_block'][0] + 1
        blocks_x = cells_x - hog_params['cells_per_block'][1] + 1
        expected_features = (blocks_y * blocks_x * 
                           hog_params['cells_per_block'][0] * 
                           hog_params['cells_per_block'][1] * 
                           hog_params['orientations'])
                           
        print(f"HOG calculation details:")
        print(f"- Image size: {gray.shape}")
        print(f"- Cells grid: {cells_y}x{cells_x}")
        print(f"- Blocks grid: {blocks_y}x{blocks_x}")
        print(f"- Expected features: {expected_features}")
        print(f"- Got features: {len(features)}")
        
        # Update params if needed
        if expected_features != params.expected_features:
            print(f"Warning: Parameters expected_features needs update to {expected_features}")
            params.expected_features = expected_features
            return features
            
        return None
            
    except Exception as e:
        print(f"Error computing HOG features: {e}")
        return None

def train_single_model(args):
    """Quiet training function for parallel processing"""
    window_size, X, y, params = args
    
    try:
        # Calculate weights
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        total = n_pos + n_neg
        weights = {1: total/(2.0*n_pos), 0: total/(2.0*n_neg)}
        
        # Normalize features first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize model with better convergence parameters
        svm = SVC(
            kernel='rbf',
            C=0.1,              # Reduced from 1.0 for better convergence
            gamma='auto',       # Changed from 'scale' to 'auto'
            class_weight=weights,
            cache_size=2000,    # Increased cache for faster convergence
            probability=False,
            random_state=42,
            verbose=False,
            tol=1e-2,          # Increased tolerance for faster convergence
            max_iter=5000,     # Reduced iterations since we improved other parameters
            shrinking=True     # Enable shrinking heuristic
        )
        
        # Train with scaled data
        svm.fit(X_scaled, y)
        
        # Save model
        pickle.dump(svm, open(os.path.join(params.dir_save_files, f'svm_model_{window_size}.pkl'), 'wb'))
        pickle.dump(scaler, open(os.path.join(params.dir_save_files, f'scaler_{window_size}.pkl'), 'wb'))
        
        return window_size, svm, scaler
        
    except Exception as e:
        return window_size, None, None

def train_unified_model(params: Parameters, training_examples_dict, train_labels_dict):
    """Parallel model training with minimal output"""
    # Print initial statistics
    print("\nTraining data statistics:")
    for window_size in params.sizes_array:
        y = train_labels_dict[window_size]
        print(f"Window {window_size}x{window_size}: {np.sum(y == 1)} positive, {np.sum(y == 0)} negative samples")
    
    # Prepare training arguments
    training_args = [(size, training_examples_dict[size], train_labels_dict[size], params) 
                    for size in params.sizes_array]
    
    # Use fewer processes to give each more resources
    num_processes = min(6, mp.cpu_count())  # Reduced from 10 to 6
    print(f"\nStarting parallel training with {num_processes} processes...")
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(train_single_model, training_args),
            total=len(params.sizes_array),
            desc="Training models"
        ))
    
    # Process results
    best_model = {}
    scalers = {}
    for window_size, model, scaler in results:
        if model is not None and scaler is not None:
            best_model[window_size] = model
            scalers[window_size] = scaler
    
    # Verify all models trained successfully
    missing_models = [size for size in params.sizes_array if size not in best_model]
    if missing_models:
        raise RuntimeError(f"Failed to train models for sizes: {missing_models}")
    
    return best_model, scalers
