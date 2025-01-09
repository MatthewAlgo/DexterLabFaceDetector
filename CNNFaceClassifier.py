import os
import torch
from PIL import Image
from torchvision import transforms
from typing import Union, Tuple
from TrainCNNFaceClassifier import ClasificatorCNN

class CNNFaceClassifier:
    def __init__(self, model_path: str):
        """
        Initialize the family member classifier with a trained model.
        
        Args:
            model_path: Path to the saved model file (.pth)
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define class labels for the family members
        self.class_labels = {
            # 0: "mom",
            # 1: "dad",
            # 2: "dexter",
            # 3: "deedee",
            # 4: "unknown"
            0: "dad",
            1: "deedee",
            2: "dexter",
            3: "mom",
            4: "unknown"
        }
        
        # Load the model architecture with correct number of classes
        self.model = ClasificatorCNN(num_classes=len(self.class_labels))
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms (same as validation transforms)
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for classification.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image_path, str):
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        return self.transform(image).unsqueeze(0)
    
    def predict(self, image_path: Union[str, Image.Image]) -> Tuple[str, float, dict]:
        """
        Classify a family member in an image and return predictions.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Tuple containing:
            - predicted_member: Name of the predicted family member
            - confidence: Confidence score for the prediction
            - all_probabilities: Dictionary of probabilities for all classes
        """
        # Preprocess the image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Get probabilities for all classes
        all_probs = probabilities[0].cpu().numpy()
        all_probabilities = {
            member: float(prob) 
            for member, prob in zip(self.class_labels.values(), all_probs)
        }
        
        # Get the predicted class label and confidence score
        predicted_member = self.class_labels[predicted_class.item()]
        confidence_score = confidence.item()
        
        return predicted_member, confidence_score, all_probabilities
    
    @classmethod
    def load_latest_model(cls, models_dir: str = 'antrenare/fisiere_salvate_algoritm'):
        """
        Load the most recent model from the specified directory.
        
        Args:
            models_dir: Directory containing the saved models
            
        Returns:
            FamilyMemberClassifier instance with the latest model loaded
        """
        # Find the latest model file
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
            
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)
        print(f"Loading model: {latest_model}")
        
        return cls(model_path)

# Example usage
def main():
    try:
        # Load the classifier with the latest model
        classifier = CNNFaceClassifier.load_latest_model()
        
        # Example single image classification
        image_path = "path/to/your/test/image.jpg"
        member, confidence, all_probs = classifier.predict(image_path)
        
        # Print results
        print(f"\nPredicted family member: {member}")
        print(f"Confidence: {confidence:.2%}")
        print("\nProbabilities for all family members:")
        for member, prob in all_probs.items():
            print(f"{member}: {prob:.2%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()