import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image # Pentru a lucra cu imagini
# Folderele si fisierele cu datele de antrenare si validare
train_directory = './train'
valid_directory = './validation'
train_csv_path = 'train.csv'
valid_csv_path = 'validation.csv'

# Punem datele pe GPU daca este disponibil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Definim transformarile ce vor fi aplicate pe imaginile din dataset
data_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Parametri de optimizare din ImageNet pentru standardizare
])
# Set de date pentru incarcarea imaginilor si etichetelor
class SetDateImagini(Dataset):
    def __init__(self, csv_file, folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nume_imagine = os.path.join(self.folder, self.data.iloc[idx, 0] + '.png')
        imag = Image.open(nume_imagine)
        label = int(self.data.iloc[idx, 1]) # Convertim eticheta in intreg
        # Convertim toate imaginile in format RGB
        if imag.mode != 'RGB':
            imag = imag.convert('RGB')
        if self.transform:
            imag = self.transform(imag)
        return imag, label
# Definim setul de date de antrenare si validare 
dataset_train = SetDateImagini(train_csv_path, train_directory, transform=data_transforms)
dataset_validare = SetDateImagini(valid_csv_path, valid_directory, transform=data_transforms)
train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=4)
valid_loader = DataLoader(dataset_validare, batch_size=128, shuffle=False, num_workers=4)

# Model cnn
class ClasificatorCNN(nn.Module):
    def __init__(self):
        super(ClasificatorCNN, self).__init__()
        self.strat_convolutie = self.strat_convo_f()
        self.strat_final = self.strat_final_f()
    def strat_convo_f(self):
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # De la 3x200x200 la 32x200x200
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # De la 32x100x100 la 64x100x100
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # De la 64x50x50 la 128x50x50
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # De la 128x25x25 la 256x25x25
            nn.ReLU(), 
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2) # 256x12x12
        )
        return model
    def strat_final_f(self):
        model = nn.Sequential(
            nn.Linear(256 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3) # 3 clase (0, 1, 2)
        )
        return model    
    def forward(self, x):
        x = self.strat_convolutie(x)
        x = x.view(x.size(0), -1) # Facem flatten la output-ul stratului de convolutie
        x = self.strat_final(x)
        return x

# Punem modelul pe GPU
model = ClasificatorCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
# Facem antrenarea modelului
def train_model(model, train_loader, valid_loader, criterion, optimizer, numar_de_epoci=100, model_path='best_model.pth'):
    cele_mai_bune_weighturi = None
    cea_mai_buna_acuratete = 0.0
    for epoca in range(numar_de_epoci): # Parcurgem fiecare epoca
        model.train()
        loss_local = 0
        # Faza de antrenare
        for imagini, labeluri in train_loader:
            imagini, labeluri = imagini.to(device), labeluri.to(device) # Mutam datele pe GPU
            optimizer.zero_grad() # Pentru a nu acumula gradientii
            outputs = model(imagini) # Calculam output-ul modelului
            loss = criterion(outputs, labeluri) # Calculam loss-ul
            loss.backward()  # Calculam gradientii
            optimizer.step() # Actualizam parametrii
            loss_local += loss.item() * imagini.size(0) # Adunam loss-ul pentru fiecare batch
        loss_per_epoca = loss_local / len(train_loader.dataset) # Calculam loss-ul mediu pentru un epoca
        print(f"Epoca de antrenare {epoca+1}/{numar_de_epoci}, Loss: {loss_per_epoca:.4f}") # Afisam loss-ul
        predictii_totale = 0
        predictii_corecte = 0
        # Faza de validare
        model.eval()        
        with torch.no_grad():
            for imagini, labeluri in valid_loader:
                imagini, labeluri = imagini.to(device), labeluri.to(device)  # Mutam datele pe GPU
                outputs = model(imagini)
                vr, prezise = torch.max(outputs.data, 1)
                predictii_totale += labeluri.size(0)
                predictii_corecte += (prezise == labeluri).sum().item()
        acuratete = predictii_corecte / predictii_totale
        print(f'Acuratete pe datele de validare: {acuratete:.4f}')

        # Salvam cel mai bun model de pana acum
        if acuratete > cea_mai_buna_acuratete:
            cea_mai_buna_acuratete = acuratete
            cele_mai_bune_weighturi = model.state_dict()
            torch.save(cele_mai_bune_weighturi, model_path)
            print(f'Cel mai bun model de pana acum: {cea_mai_buna_acuratete:.4f}')
    # Actualizam modelul cu cea mai buna configuratie
    if cele_mai_bune_weighturi:
        model.load_state_dict(cele_mai_bune_weighturi)
    return model

# plt.figure(figsize=(12, 12))
# for i in range(16):
#     image, label = dataset_train[i]
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(image.permute(1, 2, 0))
#     plt.title(f'Label: {label}')
#     plt.axis('off')
# plt.show()

# Antrenam modelul
train_model(model, train_loader, valid_loader, criterion, optimizer)