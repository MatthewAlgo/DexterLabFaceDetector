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
        
        # Calculate HOG parameters without printing
        self.window_size = 64
        test_img = np.zeros((self.window_size, self.window_size))
        self.hog_params = self._get_matching_hog_params(test_img)
        
        # Generate window sizes silently
        self.window_sizes = self._generate_window_sizes()

    def _get_matching_hog_params(self, img):
        """Fixed HOG parameters to match expected dimension of 1764"""
        base_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),  # Fixed at 8x8
            'cells_per_block': (2, 2),  # Fixed at 2x2
            'block_norm': 'L2-Hys',
            'transform_sqrt': True
        }
        
        features = hog(img, feature_vector=True, **base_params)
        if len(features) == self.feature_dim:
            return base_params
            
        # If base params don't match, try fixed alternative for 1764 features
        # 1764 = (7 blocks * 7 blocks) * (2 cells * 2 cells) * 9 orientations
        fixed_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'transform_sqrt': True
        }
        
        return fixed_params

    def compute_features(self, window):
        """Simplified feature computation with fixed parameters"""
        # Enhance contrast
        window = cv.equalizeHist(window)
        
        # Resize and compute HOG with fixed parameters
        window_resized = cv.resize(window, (64, 64))  # Fixed size
        features = hog(window_resized, feature_vector=True, **self.hog_params)
        return features
        
    def _generate_window_sizes(self, min_size=60, max_size=200, num_sizes=10):
        """Generate window sizes with variable aspect ratios"""
        widths = np.linspace(min_size, max_size, num_sizes, dtype=int)
        heights = np.linspace(min_size, max_size, num_sizes, dtype=int)
        
        # Generate all combinations
        all_combinations = list(itertools.product(widths, heights))
        
        # Filter combinations to get around 100 total windows
        # Keep windows with reasonable aspect ratios (between 0.5 and 2.0)
        filtered_sizes = []
        for w, h in all_combinations:
            aspect_ratio = w / h
            if 0.5 <= aspect_ratio <= 2.0:
                filtered_sizes.append((w, h))
        
        # If we have too many windows, sample them
        if len(filtered_sizes) > 100:
            indices = np.linspace(0, len(filtered_sizes)-1, 100, dtype=int)
            filtered_sizes = [filtered_sizes[i] for i in indices]
            
        return filtered_sizes
        
    def detect_faces(self, image):
        """Run sliding window detection using unified model"""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply preprocessing
        gray = cv.equalizeHist(gray)
        integral_img = cv.integral(gray)
        integral_sqr = cv.integral(np.float32(gray) ** 2)
        
        all_detections = []
        all_scores = []
        
        # Process each window configuration
        for window_width, window_height in self.window_sizes:
            # Adaptive stride based on window size
            stride_w = max(8, window_width // 16)
            stride_h = max(8, window_height // 16)
            
            # More strict score thresholding
            detection_threshold = self.params.threshold * 0.9  # Using 90% of threshold for initial detection
            
            # Slide window
            for y in range(0, gray.shape[0] - window_height + 1, stride_h):
                for x in range(0, gray.shape[1] - window_width + 1, stride_w):
                    # Quick variance check
                    x2, y2 = x + window_width, y + window_height
                    area = window_width * window_height
                    
                    # Calculate using integral image
                    sum_val = (integral_img[y2, x2] - integral_img[y2, x] - 
                              integral_img[y, x2] + integral_img[y, x])
                    sum_sqr = (integral_sqr[y2, x2] - integral_sqr[y2, x] - 
                              integral_sqr[y, x2] + integral_sqr[y, x])
                    
                    mean = sum_val / area
                    variance = (sum_sqr / area) - (mean ** 2)
                    
                    if variance < 100:  # Skip low-variance regions
                        continue
                        
                    # Extract and resize window to fixed size for HOG
                    window = gray[y:y2, x:x2]
                    window_resized = cv.resize(window, (64, 64))  # Fixed size for HOG
                    
                    # Get HOG features
                    features = hog(window_resized,
                                 pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                 cells_per_block=(2, 2),
                                 orientations=9,
                                 feature_vector=True)
                    
                    # Verify feature dimension
                    if len(features) != self.feature_dim:
                        continue
                    
                    # Scale features and predict
                    features_scaled = self.scaler.transform([features])
                    score = self.model.decision_function(features_scaled)[0]
                    
                    # Even stricter score filtering
                    if score >= detection_threshold and score > 3.5:  # Added minimum score requirement
                        all_detections.append([x, y, x + window_width, y + window_height])
                        all_scores.append(score)
        
        # Additional score filtering before returning
        if len(all_detections) > 0:
            detections_array = np.array(all_detections)
            scores_array = np.array(all_scores)
            high_score_mask = scores_array >= self.params.threshold
            return detections_array[high_score_mask], scores_array[high_score_mask]
        
        return np.array([]), np.array([])
        
    def visualize_all_windows(self):
        """Store visualization without displaying"""
        # Create canvas matching test image dimensions
        canvas_size = (480, 640)
        canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
        
        # Calculate center
        center_y = canvas_size[0] // 2
        center_x = canvas_size[1] // 2
        
        # Draw center point
        cv.circle(canvas, (center_x, center_y), 3, (255, 255, 255), -1)
        
        # Generate colors for different window sizes
        num_windows = len(self.window_sizes)
        colors = [
            (int(255 * (1 - i / num_windows)),  # More blue for smaller windows
             int(255 * (i / num_windows)),      # More green for larger windows
             128)                               # Constant red component
            for i in range(num_windows)
        ]
        
        # Draw each window centered
        for (w, h), color in zip(self.window_sizes, colors):
            # Calculate coordinates to center the window
            x = center_x - w // 2
            y = center_y - h // 2
            
            # Make sure window doesn't go outside canvas
            if x < 0 or y < 0 or x + w > canvas_size[1] or y + h > canvas_size[0]:
                # Scale down window if it's too large
                scale = min(
                    canvas_size[1] / w,
                    canvas_size[0] / h,
                    0.8  # Maximum scale factor to leave room for text
                )
                w_scaled = int(w * scale)
                h_scaled = int(h * scale)
                x = center_x - w_scaled // 2
                y = center_y - h_scaled // 2
                w, h = w_scaled, h_scaled
            
            # Draw rectangle
            cv.rectangle(canvas, 
                        (x, y), 
                        (x + w, y + h), 
                        color, 
                        1)  # Thinner lines
            
            # Add size text (smaller and more compact)
            text = f"{w}x{h}"
            font_scale = 0.4
            cv.putText(canvas, 
                      text,
                      (x + 2, y + 12),
                      cv.FONT_HERSHEY_SIMPLEX,
                      font_scale,
                      color,
                      1)
        
        # Add title
        title = f"Detection Windows ({len(self.window_sizes)})"
        cv.putText(canvas,
                  title,
                  (10, 20),
                  cv.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  (255, 255, 255),
                  1)
        
        # Save the visualization
        output_path = os.path.join(self.params.dir_save_files, 'detection_windows.png')
        cv.imwrite(output_path, canvas)
