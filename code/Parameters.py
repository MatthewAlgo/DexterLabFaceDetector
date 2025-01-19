import os

class Parameters:
    def __init__(self):
        # Update base paths to use ./code
        self.base_dir = '../antrenare'  # One level up from code folder
        self.dir_pos_examples = os.path.join(self.base_dir, 'exemple_pozitive_cropate')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exemple_negative_total')
        self.dir_test_examples = os.path.join('../validare/validare')
        self.path_annotations = os.path.join('../validare/task1_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'fisiere_salvate_algoritm')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        # Minimal HOG parameters for faster processing
        self.dim_hog_cell = 9      # Larger cells = fewer features
        self.cells_per_block = 3    # Minimum block size
        self.orientations = 10       # Fewer orientations
        self.block_stride = 1       # Larger stride = fewer blocks
        self.window_size = 64       # Keep base window small
        self.block_norm = 'L2-Hys'      # Fastest normalization
        self.transform_sqrt = False # Skip gamma correction

        # Skip extra processing
        self.multichannel = False
        self.gamma_correction = 1.0

        self.dim_descriptor_cell = 32
        self.number_positive_examples = 2000   # More positive examples
        self.number_negative_examples = 10000  # More negative examples
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 10000.0          # Increased detection threshold
        self.merge_overlap = 0.3        # Stricter overlap criteria

        self.use_flip_images = True  # Păstrăm flip images activat
        
        # Faster sliding window
        self.scale_factor = 0.8     # Bigger steps between scales
        # 10 window sizes with good coverage
        self.sizes_array = [
            64, 80, 96, 112, 128,
            144, 160, 176, 192, 208 
        ]
        
        # Better scale handling
        self.min_window = 64  # Smaller minimum window
        self.max_window = 200  # Larger maximum window
        
        # Visualization parameters
        self.gt_color = (0, 255, 0)     # Green for ground truth
        self.det_color = (0, 0, 255)     # Red for detections
        self.match_color = (255, 255, 0) # Yellow for matched detections
        
        # Adjusted batch size for memory efficiency
        self.batch_size = 2000  # Reduced from 6000
        self.max_descriptors_in_memory = 5000  # Reduced from 10000

        # Enhanced training parameters
        self.learning_rate = 0.001       # Smaller learning rate
        self.min_learning_rate = 0.00001 # Lower minimum
        self.momentum = 0.95             # Higher momentum
        self.validation_split = 0.15     # Less validation data
        self.random_seed = 42        # For reproducibility
        
        # Stricter detection parameters
        self.nms_threshold = 0.3     # Stricter NMS
        self.score_threshold = 0.8   # Higher score threshold

        # Recalculate expected features correctly
        cells_in_window = self.window_size // self.dim_hog_cell  # 64/8 = 8 cells
        blocks_in_window = cells_in_window - self.cells_per_block + 1  # 8-2+1 = 7 blocks
        features_per_block = self.cells_per_block * self.cells_per_block * self.orientations  # 2*2*9 = 36
        self.expected_features = blocks_in_window * blocks_in_window * features_per_block  # 7*7*36 = 1764
        
        print(f"HOG Configuration:")
        print(f"Window size: {self.window_size}x{self.window_size}")
        print(f"Cell size: {self.dim_hog_cell}x{self.dim_hog_cell}")
        print(f"Block size: {self.cells_per_block}x{self.cells_per_block} cells")
        print(f"Expected features: {self.expected_features}")
        
        print(f"Window size: {self.window_size}")
        print(f"Cells in window: {cells_in_window}")
        print(f"Blocks in window: {blocks_in_window}")
        print(f"Features per block: {features_per_block}")
        print(f"Expected total features: {self.expected_features}")