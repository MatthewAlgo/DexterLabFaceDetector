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
        self.has_cnn_model = False
        try:
            self.cnn_classifier = CNNFaceClassifier.load_latest_model()
            self.has_cnn_model = True
        except:
            self.cnn_classifier = None
    def load_classifier(self, model_file, scaler_file):
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        self.detector = SlidingWindowDetector(self.params, self.model, self.scaler)

    def _compute_hog_features(self, image):
        """Minimal HOG computation for speed"""
        try:
            gray = cv.resize(image if len(image.shape) == 2 
                            else cv.cvtColor(image, cv.COLOR_BGR2GRAY), 
                            (self.params.detection_window_size, self.params.detection_window_size))
            features = hog(gray,
                orientations=self.params.number_of_orientations,
                pixels_per_cell=(self.params.hog_cell_dimension, self.params.hog_cell_dimension),
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
        descriptor_file = os.path.join(self.params.dir_save_files,
                                     f'descriptoriExemplePozitive_{self.params.hog_cell_dimension}_'
                                     f'{self.params.number_positive_examples}_size_{window_size}.npy')
        if os.path.exists(descriptor_file):
            os.remove(descriptor_file)
        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        positive_descriptors = []
        
        print(f'Calculam descriptorii pt {len(files)} imagini pozitive pentru dimensiunea {window_size}...')
        for i, file in enumerate(files):
            try:
                img = cv.imread(file)
                if img is None:
                    continue
                 
                features = self._compute_hog_features(img)
                if features is not None:
                    positive_descriptors.append(features)
                
                if self.params.use_flip_images:
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
        descriptor_file = os.path.join(self.params.dir_save_files,
                                     f'descriptoriExempleNegative_{self.params.hog_cell_dimension}_'
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
                
                if num_rows < window_size or num_cols < window_size:
                    continue
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


    def train_classifier(self, training_examples_dict, train_labels_dict):
        self.model, self.scaler = train_unified_model(self.params, training_examples_dict, train_labels_dict)
        self.detector = SlidingWindowDetector(self.params, self.model, self.scaler)

    def run(self):
        if self.detector is None:
            raise RuntimeError("Detector not initialized. Call load_classifier first.")
            
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        all_detections = []
        all_scores = []
        all_file_names = []
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
            
            detections, scores = self.detector.find_faces_with_sliding_window(image)
            ground_truth = ground_truth_dict.get(image_name, [])
            
            if len(detections) > 0:
                detections, scores = self.non_maximal_suppression(detections, scores, image.shape[:2])
            self.show_current_image_results(image, detections, scores, ground_truth, image_name)
            if len(detections) > 0:
                all_detections.extend(detections)
                all_scores.extend(scores)
                all_file_names.extend([image_name] * len(scores))
        
        return np.array(all_detections) if all_detections else np.array([]), \
               np.array(all_scores) if all_scores else np.array([]), \
               np.array(all_file_names) if all_file_names else np.array([])

    def intersection_over_union(self, first_bounding_box, second_bounding_box):
        intersection_x_min = max(first_bounding_box[0], second_bounding_box[0])
        intersection_y_min = max(first_bounding_box[1], second_bounding_box[1])
        bounding_box_a_x_max = min(first_bounding_box[2], second_bounding_box[2])
        bounding_box_b_y_max = min(second_bounding_box[3], second_bounding_box[3])
        intersection_area = max(0, ((bounding_box_a_x_max - intersection_x_min) + 1)) * max(0, ((bounding_box_b_y_max - intersection_y_min) + 1))
        box_a_area = ((first_bounding_box[2] - first_bounding_box[0]) + 1) * ((first_bounding_box[3] - first_bounding_box[1]) + 1)
        box_b_area = ((second_bounding_box[2] - second_bounding_box[0]) + 1) * ((second_bounding_box[3] - second_bounding_box[1]) + 1)

        intersection_over_union = (intersection_area / float(((box_a_area + box_b_area) - intersection_area)))
        return intersection_over_union

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
        
        if len(image_detections) == 0:
            return np.array([]), np.array([])
            
        detected_bounding_boxes = image_detections.astype(np.float32)
        scores = image_scores.astype(np.float32)
        
        box_min_x = detected_bounding_boxes[:, 0]
        box_min_y = detected_bounding_boxes[:, 1]
        box_max_x = detected_bounding_boxes[:, 2]
        box_max_y = detected_bounding_boxes[:, 3]
        areas = (box_max_x - box_min_x + 1) * (box_max_y - box_min_y + 1)
        order = scores.argsort()[::-1]
        
        selected_indices = []
        while order.size > 0:
            i = order[0]
            selected_indices.append(i)
            overlap_min_x = np.maximum(box_min_x[i], box_min_x[order[1:]])
            overlap_minimum_y = np.maximum(box_min_y[i], box_min_y[order[1:]])
            bounding_box_max_x = np.minimum(box_max_x[i], box_max_x[order[1:]])
            bounding_box_max_y = np.minimum(box_max_y[i], box_max_y[order[1:]])
            
            intersection_width = np.maximum(0.0, bounding_box_max_x - overlap_min_x + 1)
            intersection_height = np.maximum(0.0, bounding_box_max_y - overlap_minimum_y + 1)
            inter = intersection_width * intersection_height
            intersection_over_union_ratio = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(intersection_over_union_ratio <= 0.3)[0]
            order = order[inds + 1]
        
        return detected_bounding_boxes[selected_indices].astype(np.int32), scores[selected_indices]

    
    def classify_face(self, image, bbox):
        if self.cnn_classifier is None:
            return "unknown", 0.0
        try:
            # Extract face region
            face_bbox_x1, bounding_box_y1, bounding_box_x2, bounding_box_y2 = map(int, bbox)
            face_img = image[bounding_box_y1:bounding_box_y2, face_bbox_x1:bounding_box_x2]
            face_img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_img_rgb)
            predicted_member, confidence, _ = self.cnn_classifier.predict(pil_image)
            return predicted_member, confidence
            
        except Exception as e:
            print(f"Error in face classification: {e}")
            return "unknown", 0.0
        
    def show_current_image_results(self, image, detections, scores, true_bounding_boxes, image_name, save_only=False):
        visualized_image = image.copy()
        for ground_truth_bounding_box in true_bounding_boxes:
            bounding_box_x1, bounding_box_top_y, bounding_box_x2, bounding_box_bottom_y = map(int, ground_truth_bounding_box)
            cv.rectangle(visualized_image, (bounding_box_x1, bounding_box_top_y), (bounding_box_x2, bounding_box_bottom_y), (0, 255, 0), 2)
        # Desenam dreptunghiurile gasite pentru fiecare detectie
        for detection_bounding_box, detection_confidence in zip(detections, scores):
            bounding_box_x1, bounding_box_top_y, bounding_box_x2, bounding_box_bottom_y = map(int, detection_bounding_box)    
            cv.rectangle(visualized_image, (bounding_box_x1, bounding_box_top_y), (bounding_box_x2, bounding_box_bottom_y), (0, 0, 255), 2)
            
            text_font = cv.FONT_HERSHEY_SIMPLEX
            text_font_scale = 0.5
            line_thickness = 2
            text_padding = 5
            
            if self.has_cnn_model:
                # Clasificam
                classified_face, classification_confidence = self.classify_face(image, detection_bounding_box)
                detection_score_text = f"Score: {detection_confidence:.2f}"
                face_classification_text = f"{classified_face}: {classification_confidence:.2f}"
            else:
                # Simple detection score when no CNN model
                detection_score_text = f"Score: {detection_confidence:.2f}"
                face_classification_text = "No CNN Model"
            
            # Get text sizes
            (score_w, score_h), _ = cv.getTextSize(detection_score_text, text_font, text_font_scale, line_thickness)
            (class_w, class_h), _ = cv.getTextSize(face_classification_text, text_font, text_font_scale, line_thickness)
            
            background_x1_coordinate = bounding_box_x1
            background_top_y_coordinate = bounding_box_top_y - (score_h + class_h + 3 * text_padding)
            background_bounding_box_x2 = max(bounding_box_x1 + score_w, bounding_box_x1 + class_w) + text_padding
            background_bottom_y = bounding_box_top_y
            
            visualization_overlay = visualized_image.copy()
            cv.rectangle(visualization_overlay, (background_x1_coordinate, background_top_y_coordinate), (background_bounding_box_x2, background_bottom_y), (0, 0, 0), -1)
            cv.addWeighted(visualization_overlay, 0.6, visualized_image, 0.4, 0, visualized_image)
            # Texte
            cv.putText(visualized_image, detection_score_text,
                      (bounding_box_x1 + text_padding, bounding_box_top_y - class_h - text_padding),
                      text_font, text_font_scale, (255, 255, 255),line_thickness)
            cv.putText(visualized_image, face_classification_text,(bounding_box_x1 + text_padding, bounding_box_top_y - text_padding),
                      text_font, text_font_scale, (255, 255, 255),line_thickness)
        
        if not save_only:
            window_name = f'Results_{image_name}'
            try:
                cv.imshow(window_name, visualized_image)
                while True:
                    key = cv.waitKey(100)
                    if key != -1 or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                        break
            finally:
                cv.destroyWindow(window_name)
                cv.waitKey(1)
        
        # Save img
        output_path = os.path.join(self.params.dir_save_files, f'detection_{image_name}')
        cv.imwrite(output_path, visualized_image)