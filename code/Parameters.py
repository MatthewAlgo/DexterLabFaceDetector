import os
import multiprocessing
import glob  # Add this import

class Parameters:
    def __init__(self):
        # Update paths to use relative paths correctly
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'antrenare'))
        self.dir_pos_examples = os.path.join(self.base_dir, 'exemple_pozitive_cropate')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exemple_negative_total')
        self.dir_test_examples = os.path.join(os.path.dirname(self.base_dir), 'validare', 'validare')
        self.path_annotations = os.path.join(os.path.dirname(self.base_dir), 'validare', 'task1_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'fisiere_salvate_algoritm')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # Verify directories exist
        for directory in [self.dir_pos_examples, self.dir_neg_examples]:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory not found: {directory}")
            else:
                print(f"Found directory: {directory}")
                print(f"Contains {len(glob.glob(os.path.join(directory, '*.jpg')))} images")

        # set the parameters
        self.dim_hog_cell = 8  # Increased cell size for faster processing
        self.dim_descriptor_cell = 32
        self.number_positive_examples = 5813  # Dublu față de anterior (5813 * 2)
        self.number_negative_examples = 58130  # 5813 *2 * 5 = 46448
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0.7  # Slightly lower threshold for more detections
        self.merge_overlap = 0.3  # Back to 0.3

        self.use_flip_images = True  # Păstrăm flip images activat
        
        # Updated window sizes array with 10 dimensions
        self.sizes_array = [
            80,    # Smallest window
            90,
            100,
            120,
            140,
            150,
            160,
            180,
            190,
            200    # Largest window
        ]
        
        # Dynamic window size (will be updated during processing)
        self.window_size = self.sizes_array[0]  # Start with smallest size
        
        # Window pyramid parameters
        self.scale_factor = 0.8  # Scale factor between pyramid levels
        self.min_window = 80
        self.max_window = 200
        
        # Visualization parameters
        self.gt_color = (0, 255, 0)     # Green for ground truth
        self.det_color = (0, 0, 255)     # Red for detections
        self.match_color = (255, 255, 0) # Yellow for matched detections

        # Add window size and HOG parameters
        self.sliding_window_stride = 8  # Stride for sliding window
        
        # Corrected HOG parameters for consistent dimensions
        # For 80x80 image with 10x10 pixels_per_cell:
        # - Number of cells: 8x8 (80/10)
        # - With 2x2 cells per block: 7x7 blocks (8-1)
        # - Each block has 2x2x8 = 32 features
        # Total features = 7 * 7 * 32 = 1568
        self.hog_params = {
            'orientations': 8,
            'pixels_per_cell': (10, 10),
            'cells_per_block': (2, 2),
            'block_norm': 'L2',  # Changed from L2-Hys for speed
            'transform_sqrt': True
        }
        
        # Update expected features to match actual HOG output
        self.expected_features = 1568  # 7x7 blocks * (2x2) cells * 8 orientations
        
        # Adjust other parameters for better detection
        self.threshold = 0.7  # Slightly lower threshold for more detections
        self.merge_overlap = 0.3
        
        # CNN parameters (optional)
        self.cnn_input_size = 200  # Size for CNN input images

    def set_window_size(self, size):
        """Update window size and related parameters"""
        self.window_size = size
        # Update stride based on window size
        self.sliding_window_stride = max(8, size // 16)