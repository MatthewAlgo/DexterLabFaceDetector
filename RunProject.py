from Parameters import *
from FacialDetectorDexter import *
import pdb
from Visualize import *


params: Parameters = Parameters()
# params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
# params.dim_hog_cell = 6  # dimensiunea celulei
# params.overlap = 0.3
# params.number_positive_examples = 5813  # numarul exemplelor pozitive - toate imaginile cu deedee, dexter, mom, dad
# params.number_negative_examples = 10000  # numarul exemplelor negative

# params.threshold = 2.5 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
# params.has_annotations = True

# params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
# params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetectorDexter = FacialDetectorDexter(params)

# Check if all descriptor files exist first
all_files_exist = True
descriptor_files = []

for window_size in params.sizes_array:
    pos_file = os.path.join(params.dir_save_files, 
                           f'descriptoriExemplePozitive_{params.dim_hog_cell}_'
                           f'{params.number_positive_examples}_size_{window_size}.npy')
    neg_file = os.path.join(params.dir_save_files,
                           f'descriptoriExempleNegative_{params.dim_hog_cell}_'
                           f'{params.number_negative_examples}_size_{window_size}.npy')
    descriptor_files.append((window_size, pos_file, neg_file))
    if not (os.path.exists(pos_file) and os.path.exists(neg_file)):
        all_files_exist = False
        break

# Initialize dictionaries
positive_features_dict = {}
negative_features_dict = {}

if all_files_exist:
    print("Toate fisierele cu descriptori exista. Incarcam descriptorii...")
    for window_size, pos_file, neg_file in descriptor_files:
        positive_features_dict[window_size] = np.load(pos_file)
        negative_features_dict[window_size] = np.load(neg_file)
        print(f'Am incarcat descriptorii pentru dimensiunea ferestrei {window_size}')
else:
    print("Nu toate fisierele cu descriptori exista. Generam descriptorii...")
    for window_size in params.sizes_array:
        # Generate positive descriptors
        print(f'Construim descriptorii pentru exemplele pozitive cu dimensiunea {window_size}:')
        positive_features_dict[window_size] = facial_detector.get_positive_descriptors(window_size)
        
        # Generate negative descriptors
        print(f'Construim descriptorii pentru exemplele negative cu dimensiunea {window_size}:')
        negative_features_dict[window_size] = facial_detector.get_negative_descriptors(window_size)

print("\nPregatim seturile de antrenare...")
training_examples_dict = {}
train_labels_dict = {}

for window_size in params.sizes_array:
    pos_features = positive_features_dict[window_size]
    neg_features = negative_features_dict[window_size]
    
    # Make sure we have the correct number of examples
    num_positive = len(pos_features)
    num_negative = len(neg_features)
    num_total = num_positive + num_negative
    
    print(f"Window size {window_size}:")
    print(f"  Positive examples: {num_positive}")
    print(f"  Negative examples: {num_negative}")
    
    # Adjust labels to match actual number of examples
    positive_labels = np.ones(num_positive)
    negative_labels = np.zeros(num_negative)
    
    training_examples_dict[window_size] = np.concatenate((pos_features, neg_features), axis=0)
    train_labels_dict[window_size] = np.concatenate((positive_labels, negative_labels))
    
    print(f"  Total training examples: {len(training_examples_dict[window_size])}")
    print(f"  Total labels: {len(train_labels_dict[window_size])}")

print("Incepem antrenarea clasificatorului...")
facial_detector.train_classifier(training_examples_dict, train_labels_dict)

# Pasul 5. (optional) Antrenare cu exemple puternic negative (detectii cu scor >0 din cele 274 de imagini negative)
# Daca implementati acest pas ar trebui sa modificati functia FacialDetectorDexter.run()
# astfel incat sa va returneze descriptorii detectiilor cu scor > 0 din cele 274 imagini negative
# completati codul in continuare
# TODO:  (optional)  completeaza codul in continuare


detections, scores, file_names = facial_detector.run()

if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)

# Run detection on validation set
print("\nRunning detection on validation set...")
test_files = glob.glob(os.path.join(params.dir_test_examples, '*.jpg'))

all_detections = []
all_scores = []
all_file_names = []

for image_path in test_files:
    print(f"\nProcessing {os.path.basename(image_path)}...")
    detections, scores = facial_detector.run_detection(image_path)
    
    if len(detections) > 0:
        all_detections.extend(detections)
        all_scores.extend(scores)
        all_file_names.extend([os.path.basename(image_path)] * len(scores))

# Evaluate results
if params.has_annotations:
    facial_detector.eval_detections(
        np.array(all_detections),
        np.array(all_scores),
        np.array(all_file_names)
    )