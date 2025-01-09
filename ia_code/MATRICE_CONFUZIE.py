import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def afiseaza_matrice_confuzie(labeluri_adev, labeluri_prezise, nume_clase):
    cm = confusion_matrix(labeluri_adev, labeluri_prezise)
    plt.figure(figsize=(10, 7))
    plt.title('Matrice de confuzie')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nume_clase, yticklabels=nume_clase)
    plt.xlabel('Prezis')
    plt.ylabel('Valid')
    plt.show()

mapa_labeluri_prezise = {}
with open('predictions_validation.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if rows[1] == 'label':
            continue
        mapa_labeluri_prezise[rows[0]] = int(rows[1])
mapa_labeluri_adevarate = {}
with open('validation.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if rows[1] == 'label':
            continue
        mapa_labeluri_adevarate[rows[0]] = int(rows[1])

labeluri_adev = []
labeluri_prezise = []
for key in mapa_labeluri_prezise:
    if key in mapa_labeluri_adevarate:
        labeluri_adev.append(mapa_labeluri_adevarate[key])
        labeluri_prezise.append(mapa_labeluri_prezise[key])
    else:
        print(f"{key} nu a fost gasit in fisierul de validare.")

nume_clase = ['0', '1', '2']
afiseaza_matrice_confuzie(labeluri_adev, labeluri_prezise, nume_clase)
print(classification_report(labeluri_adev, labeluri_prezise, target_names=nume_clase))