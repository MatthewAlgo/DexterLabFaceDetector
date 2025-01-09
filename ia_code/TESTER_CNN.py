import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

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
    
model = ClasificatorCNN()
def incarca_modelul(clasa_model):
    cale_model = "best_model.pth"
    model.load_state_dict(torch.load(cale_model))
    return model

def prezice_imaginea(img, model, dispozitiv='cpu'):
    # Mod evaluare
    model.eval()
    if dispozitiv == 'cuda':
        model = model.to(dispozitiv)
        img = img.to(dispozitiv)
    # Nu calcula gradientul
    with torch.no_grad():
        # Adauga o dimensiune si fa predictia
        img = img.unsqueeze(0)
        outputs = model(img)
        var, preziceri = torch.max(outputs, dim=1)
        return preziceri[0].item()
# Incarca modelul
model = incarca_modelul(ClasificatorCNN)

# Transforma datele de intrare
transformari_date = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
dataset = ImageFolder('./train', transform=transformari_date)
dir_test = "./validation"

# Pentru fiecare imagine, fa predictia
for nume_fisier in os.listdir(dir_test):
    if nume_fisier.endswith(".jpg") or nume_fisier.endswith(".jpeg") or nume_fisier.endswith(".png"):
        img = Image.open(os.path.join(dir_test, nume_fisier))
        img = img.convert('RGB')
        img_transformata = transformari_date(img)
        indice_clasa_prezisa = prezice_imaginea(img_transformata, model)
        eticheta_clasa_prezisa = dataset.classes[indice_clasa_prezisa]
        print(f"Clasa prezisa pentru '{nume_fisier}': {eticheta_clasa_prezisa}")
        with open('predictii_validare.csv', 'a') as file:
            file.write(f"{nume_fisier.split('.')[0]},{eticheta_clasa_prezisa}\n")