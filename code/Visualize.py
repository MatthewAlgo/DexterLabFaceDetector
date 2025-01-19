import cv2 as cv
import os
import numpy as np
import pdb
import ntpath
import glob
from Parameters import *


def show_detections_without_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate.
    """
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        
        indices_detections_current_image = np.where(file_names == short_file_name)[0]
        
        if len(indices_detections_current_image) > 0:
            current_detections = detections[indices_detections_current_image]
            current_scores = scores[indices_detections_current_image]
            
            # Strict threshold filter
            mask = current_scores >= 10000  # Hard-coded strict threshold
            current_detections = current_detections[mask]
            current_scores = current_scores[mask]
            
            if len(current_detections) > 0:
                # Merge overlapping detections using the new center-based approach
                current_detections, current_scores = merge_overlapping_detections(current_detections, current_scores, params.merge_overlap)
                
                # Additional score check after merging
                mask = current_scores >= params.threshold
                current_detections = current_detections[mask]
                current_scores = current_scores[mask]

                for idx, detection in enumerate(current_detections):
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, detection)
                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                    cv.putText(image, f'score:{current_scores[idx]:.2f}', 
                              (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        
        # Create unique window name
        window_name = f'Image_{short_file_name}'
        
        try:
            # Show image
            cv.imshow(window_name, image)
            print('Apasa orice tasta pentru a continua...')
            
            # Wait for key press with periodic checks
            while True:
                key = cv.waitKey(100)  # Check every 100ms
                if key != -1:  # Key was pressed
                    break
                if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                    # Window was closed
                    break
                    
        except Exception as e:
            print(f"Error showing image: {e}")
        finally:
            # Ensure window is destroyed
            cv.destroyWindow(window_name)
            cv.waitKey(1)  # Give time for window to close


def show_detections_with_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate. Deseneaza bounding box-urile prezice si cele corecte.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """

    ground_truth_bboxes = np.loadtxt(params.path_annotations, dtype='str')
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]
        
        # Strict threshold filter
        mask = current_scores >= 10000  # Hard-coded strict threshold
        current_detections = current_detections[mask]
        current_scores = current_scores[mask]

        if len(current_detections) > 0:
            # Merge overlapping detections
            current_detections, current_scores = merge_overlapping_detections(current_detections, current_scores, params.merge_overlap)
            
            # Additional score check after merging
            mask = current_scores >= params.threshold
            current_detections = current_detections[mask]
            current_scores = current_scores[mask]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]

        # show ground truth bboxes
        for detection in annotations:
            cv.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])), (0, 255, 0), thickness=1)

        cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        
        # Create unique window name
        window_name = f'Image_{short_file_name}'
        
        try:
            # Show image
            cv.imshow(window_name, image)
            print('Apasa orice tasta pentru a continua...')
            
            # Wait for key press with periodic checks
            while True:
                key = cv.waitKey(100)  # Check every 100ms
                if key != -1:  # Key was pressed
                    break
                if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                    # Window was closed
                    break
                    
        except Exception as e:
            print(f"Error showing image: {e}")
        finally:
            # Ensure window is destroyed
            cv.destroyWindow(window_name)
            cv.waitKey(1)  # Give time for window to close


def merge_overlapping_detections(detections, scores, overlap_threshold):
    """
    Unește detectiile care se suprapun într-o detecție centrată pe zona de suprapunere.
    """
    if len(detections) == 0:
        return np.array([]), np.array([])
        
    merged_detections = []
    merged_scores = []
    used = np.zeros(len(detections), dtype=bool)
    
    # Sort by score for better merging
    sort_idx = np.argsort(-scores)
    detections = detections[sort_idx]
    scores = scores[sort_idx]
    
    def get_overlap_center(box1, box2):
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    for i in range(len(detections)):
        if used[i]:
            continue
            
        current_detection = detections[i]
        current_score = scores[i]
        overlapping_boxes = [current_detection]
        overlapping_scores = [current_score]
        overlapping_indices = [i]
        
        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
                
            other_detection = detections[j]
            
            # Calculate intersection
            x1 = max(current_detection[0], other_detection[0])
            y1 = max(current_detection[1], other_detection[1])
            x2 = min(current_detection[2], other_detection[2])
            y2 = min(current_detection[3], other_detection[3])
            
            if x2 > x1 and y2 > y1:  # There is overlap
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (current_detection[2] - current_detection[0]) * (current_detection[3] - current_detection[1])
                area2 = (other_detection[2] - other_detection[0]) * (other_detection[3] - other_detection[1])
                iou = intersection / float(area1 + area2 - intersection)
                
                if iou > overlap_threshold:
                    overlapping_boxes.append(other_detection)
                    overlapping_scores.append(scores[j])
                    overlapping_indices.append(j)
                    used[j] = True
        
        if len(overlapping_boxes) > 1:
            # Calculate the center of overlap for all overlapping boxes
            centers = []
            for k in range(len(overlapping_boxes)):
                for l in range(k + 1, len(overlapping_boxes)):
                    centers.append(get_overlap_center(overlapping_boxes[k], overlapping_boxes[l]))
            
            # Use mean center
            center_x = np.mean([c[0] for c in centers])
            center_y = np.mean([c[1] for c in centers])
            
            # Calculate size for new box (average of overlapping boxes)
            avg_width = np.mean([box[2] - box[0] for box in overlapping_boxes])
            avg_height = np.mean([box[3] - box[1] for box in overlapping_boxes])
            
            # Create new centered box
            half_width = avg_width / 2
            half_height = avg_height / 2
            merged_detection = [
                center_x - half_width,
                center_y - half_height,
                center_x + half_width,
                center_y + half_height
            ]
            merged_detections.append(merged_detection)
            merged_scores.append(max(overlapping_scores))
        else:
            merged_detections.append(current_detection)
            merged_scores.append(current_score)
            
        used[i] = True
    
    merged_detections = np.array(merged_detections)
    merged_scores = np.array(merged_scores)
    
    # Final score check before returning
    if len(merged_scores) > 0:
        mask = merged_scores >= 16  # Hardcoded threshold as additional safety
        merged_detections = merged_detections[mask]
        merged_scores = merged_scores[mask]
    
    return merged_detections, merged_scores


def render_detection_window(image, detections, ground_truth, scores, params: Parameters):
    """
    Renders detection windows and ground truth boxes on the image
    """
    viz_img = image.copy()
    
    # Draw ground truth boxes in green
    for gt_box in ground_truth:
        x1, y1, x2, y2 = map(int, gt_box)
        cv.rectangle(viz_img, (x1, y1), (x2, y2), params.gt_color, 2)
    
    # Draw detection boxes with their scores
    for det, score in zip(detections, scores):
        x1, y1, x2, y2 = map(int, det)
        # Calculate IoU with ground truth boxes
        max_iou = max([intersection_over_union(det, gt) for gt in ground_truth], default=0)
        
        # Color based on IoU match
        color = params.match_color if max_iou > 0.5 else params.det_color
        
        cv.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
        cv.putText(viz_img, f'{score:.2f}', (x1, y1-5), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return viz_img

def show_detection_results(image_path, detections, scores, ground_truth, params: Parameters):
    """
    Shows and saves detection results for a single image
    """
    image = cv.imread(image_path)
    viz_img = render_detection_window(image, detections, ground_truth, scores, params)
    
    # Save visualization
    basename = os.path.basename(image_path)
    output_path = os.path.join(params.dir_save_files, f'result_{basename}')
    cv.imwrite(output_path, viz_img)
    
    # Display
    cv.imshow('Detection Results', viz_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

