import os
import shutil
import csv
def copiaza_fisiere(folder_fisier):
    current_directory = os.getcwd()
    for nume_fis, nume_folder in folder_fisier:
        path_fisier = os.path.join(current_directory, nume_fis)
        path_folder = os.path.join(current_directory, nume_folder)
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        target_path_fisier = os.path.join(path_folder, nume_fis)
        try:
            shutil.copy(path_fisier, target_path_fisier)
            print(f"Copied {nume_fis} to {nume_folder}")
        except Exception as e:
            print(f"Error copying {nume_fis} to {nume_folder}: {e}")

files_to_copy = []
with open('../validation.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        files_to_copy.append((row[0] + ".png", row[1]))
copiaza_fisiere(files_to_copy)