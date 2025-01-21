import cv2 as cv
import numpy as np
from skimage.feature import hog
from Parameters import Parameters
import itertools
import os

class SlidingWindowDetector:
    def __init__(self, params, model, scaler):
        self.params = params
        self.model = model
        self.scaler = scaler
        self.feature_dim = scaler.n_features_in_
        
        self.window_size = 64
        test_img = np.zeros((self.window_size, self.window_size))
        self.hog_params = self._get_matching_hog_params(test_img)
        
        # Generate window sizes silently
        self.window_sizes = self.calculate_variable_window_dimensions()

    def _get_matching_hog_params(self, img):
        # Hog conf
        base_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),  # Fixed at 8x8
            'cells_per_block': (2, 2),  # Fixed at 2x2
            'block_norm': 'L2-Hys',
            'transform_sqrt': True
        }
        return base_params
        
    def calculate_variable_window_dimensions(self, min_size=60, max_size=300, num_base_sizes=15):
        # Base with logarithmic spacing for small windows
        scale_based_sizes = np.logspace(np.log10(min_size), np.log10(max_size), num_base_sizes, dtype=int)
        resize_ratios = [0.5, 0.67, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.33, 1.5, 2.0]
        
        filtered_sizes = []
        for base_size in scale_based_sizes:
            for ratio in resize_ratios:
                width = int(base_size * np.sqrt(ratio))
                height = int(base_size / np.sqrt(ratio))
                
                if min_size <= width <= max_size and min_size <= height <= max_size:
                    size_tuple = (width, height)
                    if size_tuple not in filtered_sizes:
                        filtered_sizes.append(size_tuple)
        
        filtered_sizes.sort(key=lambda x: x[0] * x[1])
        
        # If more than 150 windows, sample 
        if len(filtered_sizes) > 150:
            # Keep more small windows and fewer large windows
            weights = [1.0 / (w * h) for w, h in filtered_sizes]
            weights = np.array(weights) / sum(weights)
            indices = np.random.choice(len(filtered_sizes), size=150, replace=False, p=weights)
            filtered_sizes = [filtered_sizes[i] for i in sorted(indices)]
        
        return filtered_sizes
        
    def find_faces_with_sliding_window(self, image):
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image    
        detections_list = []
        detection_scores = []
        
        # Pentru calcularea variantei
        integral_image = cv.integral(gray)
        integral_squared_image = cv.integral(np.float32(gray) ** 2)
        for window_width, window_height in self.window_sizes[::2]: 
            stride_w = max(24, window_width // 6)
            stride_h = max(24, window_height // 6)
            # Lower threshold = skip more windows
            var_threshold = 50  
            
            # Slide window
            for y in range(0, gray.shape[0] - window_height + 1, stride_h):
                for x in range(0, gray.shape[1] - window_width + 1, stride_w):
                    # Varianta
                    window_right_x, window_bottom_y = x + window_width, y + window_height
                    area = window_width * window_height
                    sum_val = (integral_image[window_bottom_y, window_right_x] - integral_image[window_bottom_y, x] - 
                              integral_image[y, window_right_x] + integral_image[y, x])
                    sum_sqr = (integral_squared_image[window_bottom_y, window_right_x] - integral_squared_image[window_bottom_y, x] - 
                              integral_squared_image[y, window_right_x] + integral_squared_image[y, x])
                    mean = sum_val / area
                    variance = (sum_sqr / area) - (mean ** 2)
                    if variance < var_threshold:
                        continue
                    
                    window = gray[y:window_bottom_y, x:window_right_x]
                    window_resized = cv.resize(window, (64, 64))
                    hog_features = hog(window_resized,
                                 pixels_per_cell=(self.params.hog_cell_dimension, self.params.hog_cell_dimension),
                                 cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                                 orientations=self.params.number_of_orientations,
                                 feature_vector=True)
                    if len(hog_features) != self.feature_dim:
                        continue
                    
                    features_scaled = self.scaler.transform([hog_features])
                    score = self.model.decision_function(features_scaled)[0]
                    if score >= self.params.threshold:
                        detections_list.append([x, y, x + window_width, y + window_height])
                        detection_scores.append(score)
        
        # Only keep detections above 10000
        if len(detections_list) > 0:
            detections_array = np.array(detections_list)
            scores_array = np.array(detection_scores)
            high_score_mask = scores_array >= self.params.threshold
            return detections_array[high_score_mask], scores_array[high_score_mask]
        
        return np.array([]), np.array([])
        