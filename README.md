# Facial Detection and Recognition System - Dexter's Laboratory

## Project Description
This project implements an automated facial detection and recognition system for characters from the animated series *Dexter's Laboratory*. The system leverages computer vision algorithms to detect faces in images and recognize the main characters: Dexter, DeeDee, Mom, and Dad.

### Objectives
- **Task 1: Facial Detection**
  - Identify all character faces in the images and highlight them with red bounding boxes.
- **Task 2: Facial Recognition**
  - Classify detected faces into four categories: Dexter, DeeDee, Mom, and Dad. Each class is marked with a specific color (blue for Dexter, yellow for DeeDee, green for Dad, purple for Mom).

---

## Dataset Structure

### Directories:
1. **`training/`**
   - Contains training data organized into the following classes: `dexter`, `deedee`, `mom`, `dad`.
   - Each directory includes 1000 annotated images with bounding boxes for character faces.

2. **`validation/`**
   - 200 images for solution validation. The structure and format are similar to the `training` directory.

3. **`testing/`**
   - Testing data for final evaluation. This data will be released during the second project phase.

4. **`evaluation/`**
   - Contains:
     - `fake_test/` - Example testing data.
     - `solution_files/` - Example format for output files.
     - `evaluation_scripts/` - Scripts to compute metrics (precision, recall).

### Annotation Format
Each annotated face corresponds to a line in the text file associated with the image, with the following format:
```
image_name xmin ymin xmax ymax character_name
```
- `xmin`, `ymin`, `xmax`, `ymax`: bounding box coordinates.
- `character_name`: `dexter`, `deedee`, `mom`, `dad`, or `unknown`.

---

## Implementation

### Constraints
1. The baseline implementation must follow this paradigm:
   - Sliding window;
   - Feature extraction (e.g., HOG, CNN, color features);
   - Classifier for predictions.
2. Advanced algorithms such as Faster-RCNN or YOLO may be used for bonus points.

### Implementation Steps
1. **Facial Detection (Task 1):**
   - Locate all faces in the images and return their coordinates and associated scores.
2. **Facial Recognition (Task 2):**
   - Classify detected faces into the target characters (Dexter, DeeDee, Mom, Dad).
3. **Evaluation:**
   - Use precision-recall curves to assess algorithm performance.

---

## Evaluation

### Performance Metrics
1. **Precision:** Percentage of correct detections out of all returned detections.
2. **Recall:** Percentage of target faces correctly detected.

### Scoring
- **Task 1:**
  - The average precision (AP) will be calculated for face detection. Full score threshold: 80% AP.
- **Task 2:**
  - Average precision for each character: Dexter, DeeDee, Mom, Dad.

---

## Grading
- **Task 1:** 4 points (facial detection).
- **Task 2:** 6 points (facial recognition, 1.5 points per character).
- **Bonus:** Advanced algorithms (e.g., YOLO, Faster-RCNN).

---

## Execution Instructions
1. **Dataset Preparation:**
   - Download and extract the dataset archive from [here](https://tinyurl.com/CAVA-2024-TEMA2).
   - Organize the data according to the specified directories.
2. **Model Training:**
   - Configure the scripts for feature extraction and classifier training.
3. **Validation:**
   - Evaluate the algorithm on data from the `validation/` directory using scripts from `evaluation_scripts/`.
4. **Final Testing:**
   - Apply the algorithm on images in the `testing/` directory and generate result files following the `solution_files/` format.

---

## Software Requirements
- Python 3.8+
- OpenCV Library
- NumPy
- Matplotlib
- Scikit-learn

---
