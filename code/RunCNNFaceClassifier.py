from torchvision import models, transforms
import torch
import torch.nn as nn
import os
import cv2 as cv
import numpy as np
from PIL import Image
import warnings

class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        self.strat_convolutie = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.strat_final = nn.Sequential(
            nn.Linear(256 * 12 * 12, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512), 
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.strat_convolutie(x)
        x = x.view(x.size(0), -1)
        x = self.strat_final(x)
        return x

class CNNFaceClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomCNN(num_classes=5)  # Changed to 5 classes
        self.model = self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),  # Match the training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['dad', 'deedee', 'dexter', 'mom', 'background']  # Added background class
        
    @staticmethod
    def load_latest_model():
        warnings.filterwarnings('ignore', category=FutureWarning)
        model_dir = os.path.join('..', 'antrenare', 'fisiere_salvate_algoritm')
        model_files = [f for f in os.listdir(model_dir) if f.startswith('model_cnn_faces_trained_epoch_')]
        if not model_files:
            raise FileNotFoundError("No trained model found")
            
        # Sort by epoch number, accuracy
        def get_model_info(filename):
            try:
                parts = filename.split('_')
                epoch = int(parts[5])
                acc = float(parts[7].replace('.pth', ''))
                return (epoch, acc)
            except:
                return (-1, -1)
            
        highest_epoch_model = max(model_files, key=get_model_info)
        trained_model_path = os.path.join(model_dir, highest_epoch_model)
        cnn_face_model = CNNFaceClassifier()
        try:
            classifier_parameters = torch.load(trained_model_path, 
                                  map_location='cpu',
                                  weights_only=True)
            cnn_face_model.model.load_state_dict(classifier_parameters)
            cnn_face_model.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
        return cnn_face_model
        
    def predict(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        with torch.no_grad():
            processed_image = self.transform(image).unsqueeze(0).to(self.device)
            outputs = self.model(processed_image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence_score, pred = torch.max(probs, 1)
            
            if pred.item() == len(self.classes) - 1: 
                return "unknown", 0.0, {}
                
            return (self.classes[pred.item()], confidence_score.item(), {class_label: class_probability.item() for class_label, class_probability in zip(self.classes[:-1], probs[0][:-1])})

    def process_image(self, image_path):
        # Load image
        image = cv.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        bounding_boxes = []
        confidence_scores = []
        predicted_labels = []
        sliding_window_size = 64 
        window_step = 32  
        resize_factor = 1.0
        minimum_window_size = 64
        
        while True:
            # Resize img
            resized_image_height, current_width = int(image.shape[0] * resize_factor), int(image.shape[1] * resize_factor)
            if resized_image_height < minimum_window_size or current_width < minimum_window_size:
                break
            resized_image = cv.resize(image, (current_width, resized_image_height))
            
            # Slide window
            for y in range(0, resized_image_height - sliding_window_size + 1, window_step):
                for x in range(0, current_width - sliding_window_size + 1, window_step):
                    window = resized_image[y:y + sliding_window_size, x:x + sliding_window_size]
                    
                    window_rgb = cv.cvtColor(window, cv.COLOR_BGR2RGB)
                    window_pil = Image.fromarray(window_rgb)
                    # Predict
                    member, confidence, _ = self.predict(window_pil)
                    if confidence > 0.7: # Conf > Threshold
                        # Back to original scale
                        bounding_box_x_start = int(x / resize_factor)
                        bounding_box_y_start = int(y / resize_factor)
                        bounding_box_x_end = int((x + sliding_window_size) / resize_factor)
                        bounding_box_y_end = int((y + sliding_window_size) / resize_factor)
                        
                        bounding_boxes.append([bounding_box_x_start, bounding_box_y_start, bounding_box_x_end, bounding_box_y_end])
                        confidence_scores.append(confidence)
                        predicted_labels.append(member)
            resize_factor *= 0.75
            
        bounding_boxes = np.array(bounding_boxes)
        confidence_scores = np.array(confidence_scores)
        
        if len(bounding_boxes) > 0:
            # Show results
            viz_img = image.copy()
            for det, score, member in zip(bounding_boxes, confidence_scores, predicted_labels):
                bounding_box_x_start, bounding_box_y_start, bounding_box_x_end, bounding_box_y_end = map(int, det)
                
                # Draw box
                cv.rectangle(viz_img, (bounding_box_x_start, bounding_box_y_start), (bounding_box_x_end, bounding_box_y_end), (0, 0, 255), 2)
                
                # Add text with background
                label = f"{member} ({score:.2f})"
                font = cv.FONT_HERSHEY_SIMPLEX
                text_font_scale = 0.5
                line_thickness = 2
                
                (label_w, label_h), _ = cv.getTextSize(label, font, text_font_scale, line_thickness)
                
                # Draw rect
                cv.rectangle(viz_img, (bounding_box_x_start, bounding_box_y_start-label_h-8), (bounding_box_x_start+label_w+4, bounding_box_y_start-4), (0, 0, 0), -1)
                cv.putText(viz_img, label, (bounding_box_x_start+2, bounding_box_y_start-8), font, text_font_scale, (255, 255, 255), line_thickness)
            
            cv.imshow('Detections', viz_img)
            print('Press any key to continue...')
            cv.waitKey(0)
            cv.destroyAllWindows()
            
            # Save
            output_path = os.path.join(os.path.dirname(image_path), 
                                     f'cnn_result_{os.path.basename(image_path)}')
            cv.imwrite(output_path, viz_img)
            
        return bounding_boxes, confidence_scores, predicted_labels