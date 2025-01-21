# Facial Detection and Recognition - Dexter's Laboratory 

## System Requirements

### Python Environment
- Python 3.8 or higher
- Virtual environment recommended

### Core Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
- OpenCV (cv2) >= 4.5.0
- NumPy >= 1.19.0
- PyTorch >= 2.0.0
- Torchvision >= 0.15.0
- scikit-image >= 0.19.0
- scikit-learn >= 1.0.0
- Pillow >= 9.0.0
- matplotlib >= 3.5.0
- tqdm >= 4.65.0
- ultralytics >= 8.0.0 (for YOLO)

### Optional Dependencies
- CUDA Toolkit >= 11.0 (for GPU acceleration)
- cuDNN >= 8.0 (for GPU acceleration)

## Project Structure
```
CAVA-2024-TEMA2/
├── antrenare/
│   ├── dexter/
│   ├── deedee/
│   ├── mom/
│   ├── dad/
│   ├── train_cnn/
│   └── fisiere_salvate_algoritm/
├── validare/
│   └── validare/
└── code/
    ├── Parameters.py
    ├── FacialDetectorDexter.py
    ├── SlidingWindowDetector.py
    ├── RunProject.py
    ├── TrainSVMModel.py
    ├── TrainCNNFaceClassifier.py
    └── RunClassifierOnData.py
```

## Quick Start

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the project:
```bash
cd code
python RunProject.py
```

## Features

### 1. Face Detection (SVM + HOG)
- Sliding window approach
- HOG feature extraction
- SVM classifier
- Non-maximum suppression

### 2. Face Recognition (CNN)
- Custom CNN architecture
- Data augmentation
- Character classification
- Confidence scoring

### 3. Performance Optimizations
- Parallel processing
- GPU acceleration (when available)
- Efficient memory management
- Adaptive window scaling

### 4. Visualization
- Real-time detection display
- Confidence scores
- Character classification
- Training progress plots

## Model Training

### SVM Training
```python
# Option 1 in the menu
# Trains HOG+SVM detector
```

### CNN Training
```python
# Option 3 in the menu
# Trains CNN classifier
```

## Additional Notes

### GPU Support
- Models automatically use CUDA if available
- Falls back to CPU if no GPU detected

### Memory Management
- Batch processing for large datasets
- Efficient descriptor caching
- Adaptive window sampling

### Performance Tips
1. Enable GPU support when possible
2. Adjust chunk_size in Parameters.py for memory constraints
3. Modify window_sizes array for speed/accuracy tradeoff

## Troubleshooting

### Common Issues
1. CUDA out of memory
   - Reduce batch sizes
   - Decrease number of windows
2. Slow processing
   - Enable GPU support
   - Adjust stride parameters
   - Reduce window count

### Error Codes
- Check logs in fisiere_salvate_algoritm/
- Verify input image dimensions
- Ensure correct data organization

## License
Academic use only. See project documentation for details.
```

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
