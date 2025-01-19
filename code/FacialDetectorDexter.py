from Parameters import *
import numpy as np
import cv2 as cv
import glob
import ntpath
from TrainSVMModel import train_unified_model
from SlidingWindowDetector import SlidingWindowDetector
import matplotlib.pyplot as plt
import pdb
import pickle
import os
from copy import deepcopy
import timeit
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from RunCNNFaceClassifier import CNNFaceClassifier

class FacialDetectorDexter:
    def __init__(self, params:Parameters):
        self.params = params
        self.model = None
        self.scaler = None
        self.detector = None
        # Initialize CNN classifier silently
        self.has_cnn_model = False
        try:
            self.cnn_classifier = CNNFaceClassifier.load_latest_model()
            self.has_cnn_model = True
        except:
            self.cnn_classifier = None

    def load_classifier(self, model_file, scaler_file):
        """Load trained model and scaler"""
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        self.detector = SlidingWindowDetector(self.params, self.model, self.scaler)

    def _compute_hog_features(self, image):
        """Minimal HOG computation for speed"""
        try:
            # Quick resize and convert
            gray = cv.resize(image if len(image.shape) == 2 
                            else cv.cvtColor(image, cv.COLOR_BGR2GRAY), 
                            (self.params.window_size, self.params.window_size))
            
            # Basic HOG with minimal parameters
            features = hog(
                gray,
                orientations=self.params.orientations,
                pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                block_norm=self.params.block_norm,
                transform_sqrt=False,
                feature_vector=True
            )
            
            return features
            
        except Exception as e:
            print(f"Error computing HOG features: {e}")
            return None

    def get_positive_descriptors(self, window_size):
        """Calculate descriptors for positive examples at given window size"""
        # Delete existing descriptor file to force regeneration
        descriptor_file = os.path.join(self.params.dir_save_files,
                                     f'descriptoriExemplePozitive_{self.params.dim_hog_cell}_'
                                     f'{self.params.number_positive_examples}_size_{window_size}.npy')
        
        if os.path.exists(descriptor_file):
            os.remove(descriptor_file)  # Force regeneration

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        positive_descriptors = []
        
        print(f'Calculam descriptorii pt {len(files)} imagini pozitive pentru dimensiunea {window_size}...')
        for i, file in enumerate(files):
            try:
                img = cv.imread(file)
                if img is None:
                    continue
                    
                # Get HOG features
                features = self._compute_hog_features(img)
                if features is not None:
                    positive_descriptors.append(features)
                
                if self.params.use_flip_images:
                    # Add flipped version
                    features_flip = self._compute_hog_features(np.fliplr(img))
                    if features_flip is not None:
                        positive_descriptors.append(features_flip)
                    
                if (i + 1) % 100 == 0:
                    print(f'Procesate {i+1}/{len(files)} imagini pozitive...')
                    
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

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
            try:
                img = cv.imread(files[i])
                if img is None:
                    continue
                    
                num_rows = img.shape[0]
                num_cols = img.shape[1]
                
                # Skip if image is too small
                if num_rows < window_size or num_cols < window_size:
                    continue
                
                # Generate random patches
                x = np.random.randint(0, num_cols - window_size, size=num_negative_per_image)
                y = np.random.randint(0, num_rows - window_size, size=num_negative_per_image)
                
                for idx in range(len(y)):
                    try:
                        patch = img[y[idx]:y[idx] + window_size, x[idx]:x[idx] + window_size]
                        features = self._compute_hog_features(patch)
                        if features is not None:
                            negative_descriptors.append(features)
                            
                    except Exception as e:
                        print(f"Error processing patch: {e}")
                        continue
                        
                if (i + 1) % 10 == 0:
                    print(f'Procesate {i+1}/{num_images} imagini negative...')
                    
            except Exception as e:
                print(f"Error processing file {files[i]}: {e}")
                continue
        
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
        """Train unified model for all window sizes"""
        self.model, self.scaler = train_unified_model(self.params, training_examples_dict, train_labels_dict)
        self.detector = SlidingWindowDetector(self.params, self.model, self.scaler)

    def run(self):
        """Run detection on test images"""
        if self.detector is None:
            raise RuntimeError("Detector not initialized. Call load_classifier first.")
            
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        all_detections = []
        all_scores = []
        all_file_names = []
        
        # Load ground truth if available
        ground_truth_dict = {}
        if self.params.has_annotations:
            try:
                annotations = np.loadtxt(self.params.path_annotations, dtype='str')
                for ann in annotations:
                    img_name = ann[0]
                    bbox = ann[1:5].astype(int)
                    if img_name not in ground_truth_dict:
                        ground_truth_dict[img_name] = []
                    ground_truth_dict[img_name].append(bbox)
            except Exception as e:
                print(f"Error loading annotations: {e}")
        
        for image_path in test_files:
            image_name = ntpath.basename(image_path)
            print(f'Processing {image_name}...')
            image = cv.imread(image_path)
            
            if image is None:
                print(f"Could not load image: {image_path}")
                continue
            
            # Run detector
            detections, scores = self.detector.detect_faces(image)
            
            # Get ground truth for current image
            ground_truth = ground_truth_dict.get(image_name, [])
            
            if len(detections) > 0:
                # Apply NMS
                detections, scores = self.non_maximal_suppression(detections, scores, image.shape[:2])
            
            # Show current results (even if no detections)
            self.show_current_image_results(image, detections, scores, ground_truth, image_name)
            
            # Add to overall results if there are detections
            if len(detections) > 0:
                all_detections.extend(detections)
                all_scores.extend(scores)
                all_file_names.extend([image_name] * len(scores))
        
        return np.array(all_detections) if all_detections else np.array([]), \
               np.array(all_scores) if all_scores else np.array([]), \
               np.array(all_file_names) if all_file_names else np.array([])

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
        
    def show_current_image_results(self, image, detections, scores, ground_truth, image_name, save_only=False):
        """Show results for current image with detections and ground truth"""
        viz_img = image.copy()
        
        # Draw ground truth boxes in green
        for gt_box in ground_truth:
            x1, y1, x2, y2 = map(int, gt_box)
            cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw detections with scores and classifications
        for det, det_score in zip(detections, scores):
            x1, y1, x2, y2 = map(int, det)
            
            # Draw bounding box
            cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Text settings
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            padding = 5
            
            # Prepare detection text
            if self.has_cnn_model:
                # Get face classification
                member, conf = self.classify_face(image, det)
                score_text = f"Score: {det_score:.2f}"
                class_text = f"{member}: {conf:.2f}"
            else:
                # Simple detection score when no CNN model
                score_text = f"Score: {det_score:.2f}"
                class_text = "No CNN Model"
            
            # Get text sizes
            (score_w, score_h), _ = cv.getTextSize(score_text, font, font_scale, thickness)
            (class_w, class_h), _ = cv.getTextSize(class_text, font, font_scale, thickness)
            
            # Calculate background rectangle positions
            bg_x1 = x1
            bg_y1 = y1 - (score_h + class_h + 3 * padding)
            bg_x2 = max(x1 + score_w, x1 + class_w) + padding
            bg_y2 = y1
            
            # Draw semi-transparent background
            overlay = viz_img.copy()
            cv.rectangle(overlay, 
                        (bg_x1, bg_y1), 
                        (bg_x2, bg_y2), 
                        (0, 0, 0), 
                        -1)
            cv.addWeighted(overlay, 0.6, viz_img, 0.4, 0, viz_img)
            
            # Draw texts
            cv.putText(viz_img, 
                      score_text,
                      (x1 + padding, y1 - class_h - padding),
                      font,
                      font_scale,
                      (255, 255, 255),
                      thickness)
            cv.putText(viz_img, 
                      class_text,
                      (x1 + padding, y1 - padding),
                      font,
                      font_scale,
                      (255, 255, 255),
                      thickness)
        
        if not save_only:
            # Show image if not save_only
            window_name = f'Results_{image_name}'
            try:
                cv.imshow(window_name, viz_img)
                while True:
                    key = cv.waitKey(100)
                    if key != -1 or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                        break
            finally:
                cv.destroyWindow(window_name)
                cv.waitKey(1)
        
        # Save visualization
        output_path = os.path.join(self.params.dir_save_files, f'detection_{image_name}')
        cv.imwrite(output_path, viz_img)