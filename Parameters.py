import os

class Parameters:
    def __init__(self):
        self.base_dir = './antrenare'
        self.dir_pos_examples = os.path.join(self.base_dir, 'exemple_pozitive_cropate')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exemple_negative_total')
        self.dir_test_examples = os.path.join('./validare/validare')  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.path_annotations = os.path.join('./validare/task1_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'fisiere_salvate_algoritm')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_hog_cell = 8  # Increased cell size for faster processing
        self.dim_descriptor_cell = 32
        self.number_positive_examples = 5813  # Dublu față de anterior (5813 * 2)
        self.number_negative_examples = 15000  # Reduced number of examples
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 6
        self.merge_overlap = 0.3

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
        
        # Window pyramid parameters
        self.scale_factor = 0.8  # Scale factor between pyramid levels
        self.min_window = 80
        self.max_window = 200
        
        # Visualization parameters
        self.gt_color = (0, 255, 0)     # Green for ground truth
        self.det_color = (0, 0, 255)     # Red for detections
        self.match_color = (255, 255, 0) # Yellow for matched detections