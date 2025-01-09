from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import os
import ntpath
from copy import deepcopy
import timeit
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from CNNFaceClassifier import CNNFaceClassifier

class FacialDetectorDexter:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None
        # Initialize CNN classifier
        try:
            self.cnn_classifier = CNNFaceClassifier.load_latest_model()
        except Exception as e:
            print(f"Error loading CNN classifier: {e}")
            self.cnn_classifier = None

    def get_positive_descriptors(self, window_size):
        """Calculate descriptors for positive examples at given window size"""
        descriptor_file = os.path.join(self.params.dir_save_files,
                                     f'descriptoriExemplePozitive_{self.params.dim_hog_cell}_'
                                     f'{self.params.number_positive_examples}_size_{window_size}.npy')
        
        if os.path.exists(descriptor_file):
            return np.load(descriptor_file)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        positive_descriptors = []
        
        print(f'Calculam descriptorii pt {len(files)} imagini pozitive pentru dimensiunea {window_size}...')
        for i, file in enumerate(files):
            img = cv.imread(file)
            if img is None:
                continue
                
            # Convert to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Resize to window size
            resized = cv.resize(gray, (window_size, window_size))
            
            # Extract HOG features
            features = hog(resized, 
                          pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                          cells_per_block=(2, 2), 
                          feature_vector=True)
            positive_descriptors.append(features)
            
            if self.params.use_flip_images:
                # Add flipped version
                features_flip = hog(np.fliplr(resized), 
                                  pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                  cells_per_block=(2, 2), 
                                  feature_vector=True)
                positive_descriptors.append(features_flip)
                
            if (i + 1) % 100 == 0:
                print(f'Procesate {i+1}/{len(files)} imagini pozitive...')

        positive_descriptors = np.array(positive_descriptors)
        np.save(descriptor_file, positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self, window_size):
        """Calculate descriptors for negative examples at given window size"""
        descriptor_file = os.path.join(self.params.dir_save_files,
                                     f'descriptoriExempleNegative_{self.params.dim_hog_cell}_'
                                     f'{self.params.number_negative_examples}_size_{window_size}.npy')
        
        if os.path.exists(descriptor_file):
            return np.load(descriptor_file)
            
        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        num_negative_per_image = self.params.number_negative_examples // num_images
        negative_descriptors = []
        
        print(f'Calculam descriptorii pt {num_images} imagini negative pentru dimensiunea {window_size}...')
        for i in range(num_images):
            print(f'Procesam exemplul negativ numarul {i}/{num_images}...')
            img = cv.imread(files[i])
            if img is None:
                continue
                
            num_rows = img.shape[0]
            num_cols = img.shape[1]
            
            # Skip if image is too small
            if num_rows <= window_size or num_cols <= window_size:
                img = cv.resize(img, (window_size * 2, window_size * 2))
                num_rows, num_cols = img.shape[:2]
            
            # Generate random patches
            x = np.random.randint(low=0, high=num_cols - window_size, size=num_negative_per_image)
            y = np.random.randint(low=0, high=num_rows - window_size, size=num_negative_per_image)
            
            for idx in range(len(y)):
                patch = img[y[idx]:y[idx] + window_size, x[idx]:x[idx] + window_size]
                gray_patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
                features = hog(gray_patch,
                             pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                             cells_per_block=(2, 2),
                             feature_vector=True)
                negative_descriptors.append(features)
                
            if (i + 1) % 10 == 0:
                print(f'Procesate {i+1}/{num_images} imagini negative...')
        
        negative_descriptors = np.array(negative_descriptors)
        np.save(descriptor_file, negative_descriptors)
        return negative_descriptors

    def get_window_pyramid(self, image, min_size=None, max_size=None):
        """
        Generates windows of different sizes for the image
        """
        if (min_size is None):
            min_size = self.params.min_window
        if (max_size is None):
            max_size = self.params.max_window
            
        windows = []
        current_size = min_size
        
        while (current_size <= max_size):
            windows.append(current_size)
            current_size = int((current_size / self.params.scale_factor))
        
        return windows

    def train_classifier(self, training_examples_dict, train_labels_dict):
        """Enhanced classifier training with better model caching"""
        self.best_model = {}
        self.scalers = {}
        training_needed = False
        
        # First check all models
        print("\nVerificăm modelele existente...")
        for window_size in self.params.sizes_array:
            model_file = os.path.join(self.params.dir_save_files, f'svm_model_{window_size}.pkl')
            scaler_file = os.path.join(self.params.dir_save_files, f'scaler_{window_size}.pkl')
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                try:
                    print(f'Încărcăm modelul existent pentru window_size={window_size}')
                    self.best_model[window_size] = pickle.load(open(model_file, 'rb'))
                    self.scalers[window_size] = pickle.load(open(scaler_file, 'rb'))
                    print(f'Model încărcat cu succes pentru window_size={window_size}')
                except Exception as e:
                    print(f'Eroare la încărcarea modelului pentru window_size={window_size}: {e}')
                    training_needed = True
            else:
                print(f'Model inexistent pentru window_size={window_size}')
                training_needed = True
        
        # If any model is missing or corrupted, train all models for consistency
        if training_needed:
            print("\nUnele modele lipsesc sau sunt corupte. Se antrenează modele noi...")
            for window_size in self.params.sizes_array:
                print(f'\nAntrenăm model pentru window_size={window_size}')
                X = training_examples_dict[window_size]
                y = train_labels_dict[window_size]
                
                # Use stronger linear SVM with balanced class weights
                svm = LinearSVC(
                    C=1.0,
                    class_weight='balanced',
                    max_iter=10000,
                    tol=1e-5,
                    dual=False
                )
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                try:
                    svm.fit(X_scaled, y)
                    self.best_model[window_size] = svm
                    self.scalers[window_size] = scaler
                    
                    # Save model and scaler
                    model_file = os.path.join(self.params.dir_save_files, f'svm_model_{window_size}.pkl')
                    scaler_file = os.path.join(self.params.dir_save_files, f'scaler_{window_size}.pkl')
                    
                    print(f'Salvăm modelul pentru window_size={window_size}')
                    pickle.dump(svm, open(model_file, 'wb'))
                    pickle.dump(scaler, open(scaler_file, 'wb'))
                    
                    print(f'Model antrenat și salvat cu succes pentru window_size={window_size}')
                    
                except Exception as e:
                    print(f'Eroare la antrenare pentru window_size={window_size}: {e}')
                    raise  # Re-raise exception as we need all models
        else:
            print("\nToate modelele există și au fost încărcate cu succes!")

        # Verify we have all models
        missing_models = [size for size in self.params.sizes_array if size not in self.best_model]
        if missing_models:
            raise RuntimeError(f"Lipsesc modele pentru dimensiunile: {missing_models}")

    def run_detection(self, image_path):
        """
        Runs detection on a single image and shows results
        """
        image = cv.imread(image_path)
        detections = []
        scores = []
        
        # Get ground truth for this image
        gt_boxes = self.get_ground_truth_boxes(image_path)
        
        # Run detection at multiple scales
        for window_size in self.params.sizes_array:
            dets, scrs = self.detect_faces_at_scale(image, window_size, self.best_model[window_size])
            if (len(dets) > 0):
                detections.extend(dets)
                scores.extend(scrs)
        
        # Apply NMS
        if (len(detections) > 0):
            detections = np.array(detections)
            scores = np.array(scores)
            detections, scores = self.non_maximal_suppression(detections, scores, image.shape[:2])
            
            # Show results
            show_detection_results(image_path, detections, scores, gt_boxes, self.params)
        
        return detections, scores

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_b[3], bbox_b[3])

        inter_area = max(0, ((x_b - x_a) + 1)) * max(0, ((y_b - y_a) + 1))

        box_a_area = ((bbox_a[2] - bbox_a[0]) + 1) * ((bbox_a[3] - bbox_a[1]) + 1)
        box_b_area = ((bbox_b[2] - bbox_b[0]) + 1) * ((bbox_b[3] - bbox_b[1]) + 1)

        iou = (inter_area / float(((box_a_area + box_b_area) - inter_area)))

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        # x_out_of_bounds = np.where((image_detections[:, 2] > image_size[1]))[0]
        # y_out_of_bounds = np.where((image_detections[:, 3] > image_size[0]))[0]
        # print(x_out_of_bounds, y_out_of_bounds)
        # image_detections[x_out_of_bounds, 2] = image_size[1]
        # image_detections[y_out_of_bounds, 3] = image_size[0]
        # sorted_indices = np.flipud(np.argsort(image_scores))
        # sorted_image_detections = image_detections[sorted_indices]
        # sorted_scores = image_scores[sorted_indices]

        # is_maximal = np.ones(len(image_detections)).astype(bool)
        # iou_threshold = 0.3
        # for i in range((len(sorted_image_detections) - 1)):
        #     if (is_maximal[i] == True):  # don't change to 'is True' because is a numpy True and is not a python True :)
        #         for j in range((i + 1), len(sorted_image_detections)):
        #             if (is_maximal[j] == True):  # don't change to 'is True' because is a numpy True and is not a python True :)
        #                 if (self.intersection_over_union(sorted_image_detections[i], sorted_image_detections[j]) > iou_threshold):
        #                     is_maximal[j] = False
        #                 else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
        #                     c_x = ((sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2)
        #                     c_y = ((sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2)
        #                     if ((sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2]) and (sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3])):
        #                         is_maximal[j] = False
        # return (sorted_image_detections[is_maximal], sorted_scores[is_maximal])
        
        """Improved NMS with better overlap handling"""
        if len(image_detections) == 0:
            return np.array([]), np.array([])
            
        # Convert to float for better precision
        boxes = image_detections.astype(np.float32)
        scores = image_scores.astype(np.float32)
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            # Calculate IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Get indices of boxes with IoU <= threshold
            inds = np.where(ovr <= 0.3)[0]
            order = order[inds + 1]
        
        return boxes[keep].astype(np.int32), scores[keep]

    def run(self):
        """Updated run method with immediate visualization"""
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        all_detections = []
        all_scores = []
        all_file_names = []
        
        # Load ground truth if available
        if self.params.has_annotations:
            ground_truth = np.loadtxt(self.params.path_annotations, dtype='str')
        
        for i, test_file in enumerate(test_files):
            start_time = timeit.default_timer()
            print(f'Procesam imaginea de testare {i+1}/{len(test_files)}...')
            
            image = cv.imread(test_file)
            image_detections = []
            image_scores = []
            
            # Process each window size
            for window_size in self.params.sizes_array:
                model = self.best_model[window_size]
                scaler = self.scalers[window_size]
                
                # Process image at different scales
                scale = 1.0
                current_img = image.copy()
                
                while current_img.shape[0] >= window_size and current_img.shape[1] >= window_size:
                    dets, scores = self.detect_faces_at_scale(current_img, window_size, model, scaler)
                    
                    if len(dets) > 0:
                        dets = (dets / scale).astype(np.int32)
                        image_detections.extend(dets)
                        image_scores.extend(scores)
                    
                    scale *= 0.75
                    current_img = cv.resize(image, None, fx=scale, fy=scale)
            
            # Process detections for current image
            if len(image_scores) > 0:
                image_detections = np.array(image_detections)
                image_scores = np.array(image_scores)
                
                # Filter by threshold
                mask = image_scores >= self.params.threshold
                image_detections = image_detections[mask]
                image_scores = image_scores[mask]
                
                if len(image_scores) > 0:
                    # Apply NMS
                    final_dets, final_scores = self.non_maximal_suppression(
                        image_detections, image_scores, image.shape[:2])
                    
                    # Add to global lists
                    all_detections.extend(final_dets)
                    all_scores.extend(final_scores)
                    current_file_name = ntpath.basename(test_file)
                    all_file_names.extend([current_file_name] * len(final_scores))
                    
                    # Get ground truth for current image
                    if self.params.has_annotations:
                        current_gt = ground_truth[ground_truth[:, 0] == current_file_name]
                        gt_boxes = current_gt[:, 1:5].astype(int) if len(current_gt) > 0 else np.array([])
                    else:
                        gt_boxes = np.array([])
                    
                    # Show results for current image
                    viz_img = self.visualize_current_image(
                        image, 
                        final_dets, 
                        final_scores, 
                        gt_boxes,
                        f"Image {i+1}/{len(test_files)}"
                    )
                    
                    # Save visualization
                    output_path = os.path.join(self.params.dir_save_files, 
                                             f'detection_{current_file_name}')
                    cv.imwrite(output_path, viz_img)
            
            end_time = timeit.default_timer()
            print(f'Timpul de procesare: {end_time - start_time:.2f} sec.')
            
            # Show results for current image
            current_file_name = ntpath.basename(test_file)
            if self.params.has_annotations:
                current_gt = ground_truth[ground_truth[:, 0] == current_file_name]
                gt_boxes = current_gt[:, 1:5].astype(int) if len(current_gt) > 0 else np.array([])
            else:
                gt_boxes = np.array([])
                
            print("Afișez rezultatele detecției pentru imaginea curentă...")
            self.show_current_image_results(
                image,
                final_dets if 'final_dets' in locals() else np.array([]),
                final_scores if 'final_scores' in locals() else np.array([]),
                gt_boxes,
                current_file_name
            )
        
        return np.array(all_detections), np.array(all_scores), np.array(all_file_names)

    def visualize_current_image(self, image, detections, scores, ground_truth, title):
        """Visualize detections and ground truth for current image"""
        viz_img = image.copy()
        
        # Draw ground truth boxes in green
        for gt_box in ground_truth:
            x1, y1, x2, y2 = map(int, gt_box)
            cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw detections with scores
        for det, score in zip(detections, scores):
            x1, y1, x2, y2 = map(int, det)
            color = (0, 0, 255)  # Red for detections
            cv.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
            cv.putText(viz_img, 
                      f'{score:.2f}', 
                      (x1, y1-5),
                      cv.FONT_HERSHEY_SIMPLEX, 
                      0.5, 
                      color, 
                      2)
        
        # Show image
        cv.imshow(title, viz_img)
        cv.waitKey(1)  # Short delay to show image
        
        return viz_img

    def detect_faces_at_scale(self, image, window_size, model, scaler):
        """Improved sliding window detection with adaptive stride and pruning"""
        detections = []
        scores = []
        
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply preprocessing
        gray = cv.equalizeHist(gray)
        
        # Compute integral image for faster mean/std calculations
        integral_img = cv.integral(gray)
        integral_sqr = cv.integral(np.float32(gray) ** 2)
        
        # Adaptive stride based on image size
        base_stride = self.params.dim_hog_cell
        h, w = gray.shape
        stride = max(base_stride, int(min(w, h) * 0.02))  # Adaptive stride
        
        # Get expected feature dimension
        expected_dim = scaler.n_features_in_
        
        # Slide window with pruning
        for y in range(0, gray.shape[0] - window_size + 1, stride):
            for x in range(0, gray.shape[1] - window_size + 1, stride):
                # Quick variance check to skip low-information regions
                x2, y2 = x + window_size, y + window_size
                area = window_size * window_size
                
                # Calculate mean and variance using integral image
                sum_val = (integral_img[y2, x2] - integral_img[y2, x] - 
                          integral_img[y, x2] + integral_img[y, x])
                sum_sqr = (integral_sqr[y2, x2] - integral_sqr[y2, x] - 
                          integral_sqr[y, x2] + integral_sqr[y, x])
                
                mean = sum_val / area
                variance = (sum_sqr / area) - (mean ** 2)
                
                # Skip low-variance regions (likely background)
                if variance < 100:  # Threshold determined empirically
                    continue
                
                # Extract window
                window = gray[y:y + window_size, x:x + window_size]
                
                # Get features
                features = self.get_combined_descriptor(window, window_size)
                
                # Skip if feature dimension doesn't match
                if len(features) != expected_dim:
                    continue
                
                # Scale features
                features_scaled = scaler.transform([features])
                
                # Get prediction score
                score = model.decision_function(features_scaled)[0]
                
                if score >= self.params.threshold:
                    # Add detection with confidence score
                    detections.append([x, y, x + window_size, y + window_size])
                    scores.append(score)
        
        return np.array(detections), np.array(scores)

    def get_combined_descriptor(self, image, window_size):
        """Get combined HOG and color features for a window"""
        # Store feature dimensions if we haven't yet
        if not hasattr(self, 'feature_dimensions'):
            self.feature_dimensions = {}
        
        # Resize first to reduce computation
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize to window size if needed
        if gray.shape != (window_size, window_size):
            gray = cv.resize(gray, (window_size, window_size))
        
        # Get HOG features with fixed parameters
        hog_features = hog(gray, 
                          pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                          cells_per_block=(2, 2),
                          orientations=9,
                          feature_vector=True)
        
        # Just use HOG features for consistency
        features = hog_features
        
        # Store feature dimension for this window size
        self.feature_dimensions[window_size] = len(features)
        
        return features

    def get_hog_descriptor(self, img):
        """Calculate HOG features"""
        return hog(
            img,
            orientations=12,  # Increased from 9 to better capture facial features
            pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
            cells_per_block=(3, 3),  # Increased from 2x2 for better normalization
            feature_vector=True,
            block_norm='L2-Hys',
            transform_sqrt=True  # Apply gamma correction for better contrast
        )

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
    
    def classify_face(self, image, bbox):
        """
        Classify a detected face using the CNN classifier
        
        Args:
            image: Full image (OpenCV/numpy format)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
        
        Returns:
            tuple: (predicted_member, confidence)
        """
        if self.cnn_classifier is None:
            return "unknown", 0.0
            
        try:
            # Extract face region
            x1, y1, x2, y2 = map(int, bbox)
            face_img = image[y1:y2, x1:x2]
            
            # Convert from OpenCV to PIL format
            # OpenCV uses BGR, PIL uses RGB
            face_img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_img_rgb)
            
            # Use CNN classifier to predict
            predicted_member, confidence, _ = self.cnn_classifier.predict(pil_image)
            return predicted_member, confidence
            
        except Exception as e:
            print(f"Error in face classification: {e}")
            return "unknown", 0.0
        
    # def show_current_image_results(self, image, detections, scores, ground_truth, image_name):
    #     """Show results for current image with detections and ground truth"""
    #     viz_img = image.copy()
        
    #     # Draw ground truth boxes in green
    #     for gt_box in ground_truth:
    #         x1, y1, x2, y2 = map(int, gt_box)
    #         cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    #     # Draw detections with scores in red
    #     for det, score in zip(detections, scores):
    #         x1, y1, x2, y2 = map(int, det)
    #         cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #         cv.putText(viz_img, 
    #                   f'score: {score:.2f}', 
    #                   (x1, y1-5),
    #                   cv.FONT_HERSHEY_SIMPLEX, 
    #                   0.5, 
    #                   (0, 0, 255), 
    #                   2)
        
    #     # Show image
    #     cv.imshow(f'Results for {image_name}', viz_img)
    #     print('Apasă orice tastă pentru a continua...')
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
        
    #     # Save visualization
    #     output_path = os.path.join(self.params.dir_save_files, f'detection_{image_name}')
    #     cv.imwrite(output_path, viz_img)

    def show_current_image_results(self, image, detections, scores, ground_truth, image_name):
        """Show results for current image with detections, classifications and ground truth"""
        viz_img = image.copy()
        
        # Draw ground truth boxes in green
        for gt_box in ground_truth:
            x1, y1, x2, y2 = map(int, gt_box)
            cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw detections with scores and classifications in red
        for det, det_score in zip(detections, scores):
            x1, y1, x2, y2 = map(int, det)
            
            # Get face classification
            member, conf = self.classify_face(image, det)
            
            # Draw bounding box
            cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw detection score and classification
            label = f"{member} ({conf:.2f})"
            det_text = f"det: {det_score:.2f}"
            
            # Add text with background for better visibility
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # Get text sizes
            (label_w, label_h), _ = cv.getTextSize(label, font, font_scale, thickness)
            (det_w, det_h), _ = cv.getTextSize(det_text, font, font_scale, thickness)
            
            # Draw background rectangles
            cv.rectangle(viz_img, 
                        (x1, y1-label_h-8), 
                        (x1+label_w+4, y1-4), 
                        (0, 0, 0), 
                        -1)
            cv.rectangle(viz_img, 
                        (x1, y1-label_h-det_h-12), 
                        (x1+det_w+4, y1-label_h-8), 
                        (0, 0, 0), 
                        -1)
            
            # Draw text
            cv.putText(viz_img, 
                      label,
                      (x1+2, y1-8),
                      font,
                      font_scale,
                      (255, 255, 255),
                      thickness)
            cv.putText(viz_img, 
                      det_text,
                      (x1+2, y1-label_h-10),
                      font,
                      font_scale,
                      (255, 255, 255),
                      thickness)
        
        # Show image
        cv.imshow(f'Results for {image_name}', viz_img)
        print('Apasă orice tastă pentru a continua...')
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        # Save visualization
        output_path = os.path.join(self.params.dir_save_files, f'detection_{image_name}')
        cv.imwrite(output_path, viz_img)
