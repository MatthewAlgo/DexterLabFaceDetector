from torchvision import models, transforms
import torch
import torch.nn as nn
import os
import cv2 as cv
import numpy as np
from PIL import Image

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
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # New layer
            nn.ReLU(),
            nn.BatchNorm2d(256)  # New layer
        )
        
        self.strat_final = nn.Sequential(
            nn.Linear(256 * 12 * 12, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),  # Added batch norm
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # New intermediate layer
            nn.ReLU(),
            nn.Dropout(0.3),  # Lower dropout for deeper layer
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
        """Load the latest trained model silently"""
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        model_dir = os.path.join('..', 'antrenare', 'fisiere_salvate_algoritm')
        model_files = [f for f in os.listdir(model_dir) if f.startswith('model_cnn_faces_trained_epoch_')]
        
        if not model_files:
            raise FileNotFoundError("No trained model found")
            
        # Sort by epoch number and accuracy
        def get_model_info(filename):
            try:
                parts = filename.split('_')
                epoch = int(parts[5])
                acc = float(parts[7].replace('.pth', ''))
                return (epoch, acc)
            except:
                return (-1, -1)
            
        latest_model = max(model_files, key=get_model_info)
        model_path = os.path.join(model_dir, latest_model)
        
        classifier = CNNFaceClassifier()
        
        try:
            # Load model on CPU
            state_dict = torch.load(model_path, 
                                  map_location='cpu',
                                  weights_only=True)
            classifier.model.load_state_dict(state_dict)
            classifier.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
        return classifier
        
    def predict(self, image):
        """Predict face class from image"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        # Transform and predict
        with torch.no_grad():
            x = self.transform(image).unsqueeze(0).to(self.device)
            outputs = self.model(x)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
            # Only return face class predictions (ignore background)
            if pred.item() == len(self.classes) - 1:  # If background
                return "unknown", 0.0, {}
                
            return (self.classes[pred.item()], 
                    conf.item(),
                    {c: p.item() for c, p in zip(self.classes[:-1], probs[0][:-1])})

    def process_image(self, image_path):
        """Process an image and show detections with classifications"""
        image = cv.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
            
        # Store detections and scores
        detections = []
        scores = []
        classifications = []
        
        # Process each sliding window
        window_size = 64  # Base window size
        stride = 32      # Stride size
        
        scale = 1.0
        min_size = 64
        
        while True:
            # Resize image
            current_h, current_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
            if current_h < min_size or current_w < min_size:
                break
                
            resized = cv.resize(image, (current_w, current_h))
            
            # Slide window
            for y in range(0, current_h - window_size + 1, stride):
                for x in range(0, current_w - window_size + 1, stride):
                    # Extract window
                    window = resized[y:y + window_size, x:x + window_size]
                    
                    # Convert to PIL
                    window_rgb = cv.cvtColor(window, cv.COLOR_BGR2RGB)
                    window_pil = Image.fromarray(window_rgb)
                    
                    # Get prediction
                    member, confidence, _ = self.predict(window_pil)
                    
                    # If confidence is high enough
                    if confidence > 0.7:  # Threshold
                        # Convert coordinates back to original scale
                        x1 = int(x / scale)
                        y1 = int(y / scale)
                        x2 = int((x + window_size) / scale)
                        y2 = int((y + window_size) / scale)
                        
                        detections.append([x1, y1, x2, y2])
                        scores.append(confidence)
                        classifications.append(member)
            
            # Update scale for next pyramid level
            scale *= 0.75
            
        # Convert to numpy arrays
        detections = np.array(detections)
        scores = np.array(scores)
        
        if len(detections) > 0:
            # Show results
            viz_img = image.copy()
            for det, score, member in zip(detections, scores, classifications):
                x1, y1, x2, y2 = map(int, det)
                
                # Draw box
                cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add text with background
                label = f"{member} ({score:.2f})"
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                
                # Get text size
                (label_w, label_h), _ = cv.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle
                cv.rectangle(viz_img, 
                           (x1, y1-label_h-8), 
                           (x1+label_w+4, y1-4), 
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
            
            # Show image
            cv.imshow('Detections', viz_img)
            print('Press any key to continue...')
            cv.waitKey(0)
            cv.destroyAllWindows()
            
            # Save result
            output_path = os.path.join(os.path.dirname(image_path), 
                                     f'cnn_result_{os.path.basename(image_path)}')
            cv.imwrite(output_path, viz_img)
            
        return detections, scores, classifications