import os

class Parameters:
    def __init__(self):
        
        self.base_dir = '../antrenare'  
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
        
        self.hog_cell_dimension = 9      
        self.cells_per_block = 3    
        self.number_of_orientations = 10       
        self.block_stride = 1       
        self.detection_window_size = 64       
        self.block_norm = 'L2-Hys' 

        self.number_positive_examples = 2000   
        self.number_negative_examples = 10000  
        self.has_annotations = False
        self.threshold = 10000.0          
        self.merge_overlap = 0.3        

        self.use_flip_images = True  
        
        self.scale_factor = 0.8    
        self.sizes_array = [
            64, 80, 96, 112, 128,
            144, 160, 176, 192, 208 
        ]
        
        self.min_window = 64  
        self.max_window = 200   
        
        cells_in_detection_window = self.detection_window_size // self.hog_cell_dimension  
        total_blocks_in_window = cells_in_detection_window - self.cells_per_block + 1  
        num_features_in_block = self.cells_per_block * self.cells_per_block * self.number_of_orientations  
        self.computed_feature_count = total_blocks_in_window * total_blocks_in_window * num_features_in_block  
        
        print(f"HOG Configuration:")
        print(f"Window size: {self.detection_window_size}x{self.detection_window_size}")
        print(f"Cell size: {self.hog_cell_dimension}x{self.hog_cell_dimension}")
        print(f"Block size: {self.cells_per_block}x{self.cells_per_block} cells")
        print(f"Expected features: {self.computed_feature_count}")
        
        print(f"Window size: {self.detection_window_size}")
        print(f"Cells in window: {cells_in_detection_window}")
        print(f"Blocks in window: {total_blocks_in_window}")
        print(f"Features per block: {num_features_in_block}")
        print(f"Expected total features: {self.computed_feature_count}")