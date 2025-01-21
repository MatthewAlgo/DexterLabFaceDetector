import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create save directory if it doesn't exist
save_dir = os.path.join('../antrenare', 'fisiere_salvate_algoritm')
os.makedirs(save_dir, exist_ok=True)

# Data augmentation and transforms
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # Added blur augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modified CNN for 5 classes
class ClasificatorCNN(nn.Module):
    def __init__(self, num_classes=5):  # Modified for 5 classes
        super(ClasificatorCNN, self).__init__()
        self.strat_convolutie = self.strat_convo_f()
        self.strat_final = self.strat_final_f(num_classes)
    
    def strat_convo_f(self):
        model = nn.Sequential(
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
        return model
    
    def strat_final_f(self, num_classes):
        model = nn.Sequential(
            nn.Linear(256 * 12 * 12, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, num_classes)
        )
        return model
    
    def forward(self, x):
        x = self.strat_convolutie(x)
        x = x.view(x.size(0), -1)
        x = self.strat_final(x)
        return x

def save_model(model, epoch, accuracy, save_dir):
    """Save model with proper metadata"""
    os.makedirs(save_dir, exist_ok=True)
    model_name = f'model_cnn_faces_trained_epoch_{epoch}_acc_{accuracy:.4f}.pth'
    model_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=True)
    print(f"Model saved to {model_path}")

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50):
    model = model.to(device)
    best_weights = None
    best_accuracy = 0.0
    best_epoch = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.4f}')
        
        # Save model with descriptive name if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = model.state_dict()
            best_epoch = epoch + 1
            save_model(model, best_epoch, accuracy, save_dir)
            print(f'New best model saved! Accuracy: {best_accuracy:.4f}')
    
    # Load best model
    if best_weights:
        model.load_state_dict(best_weights)
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()
    
    return model, best_epoch, best_accuracy

def main():
    # Load dataset
    full_dataset = datasets.ImageFolder(
        root='../antrenare/fisiere_salvate_algoritm/train_cnn',
        transform=train_transforms
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Override validation transform
    val_dataset.dataset.transform = val_transforms
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model, loss function, and optimizer
    model = ClasificatorCNN(num_classes=len(full_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    trained_model, best_epoch, best_accuracy = train_model(model, train_loader, valid_loader, criterion, optimizer)
    
    print(f"Training complete! Best model saved at epoch {best_epoch} with accuracy {best_accuracy:.4f}")
    print(f"Class mapping: {full_dataset.class_to_idx}")

if __name__ == "__main__":
    main()