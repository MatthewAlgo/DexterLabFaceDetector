import os
import shutil
import cv2 as cv

def creeazaDatasetTotal():
    foldere_personaje = ['dad', 'mom', 'deedee', 'dexter']
    basepath = 'antrenare'
    folder_destinatie = os.path.join(basepath, 'exemple_pozitive_total')
    
    os.makedirs(folder_destinatie, exist_ok=True)
    for personaj in foldere_personaje:
        cale_sursa = os.path.join(basepath, personaj)
        for numar_imagine in range(1, 1001):
            numar_imagine_formatat = str(numar_imagine).zfill(4)
            file_source = os.path.join(cale_sursa, f'{numar_imagine_formatat}.jpg')
            file_dest = os.path.join(folder_destinatie, f'{numar_imagine_formatat}_{personaj}.jpg')
            if os.path.exists(file_source):
                shutil.copy2(file_source, file_dest)
            else:
                print(f"Atentie: Fisierul {file_source} nu a fost gasit")
                
def citesteAdnotarilePentruFiecareImagine(nume_personaj="dexter"):
    file_path = os.path.join('antrenare', f'{nume_personaj}_annotations.txt')
    with open(file_path, 'r') as f:
        continut = f.readlines()
    adnotari = {}
    for linie in continut:
        nume_imagine, x_min, y_min, x_max, y_max, personaj = linie.split()
        if nume_imagine not in adnotari:
            adnotari[nume_imagine] = []
        adnotari[nume_imagine].append({'coords': (int(x_min), int(y_min), int(x_max), int(y_max)), 'personaj': personaj
        })
    return adnotari

def afiseazaFiecareImagineCuBoundingBoxes(nume_personaj="dexter"):
    adnotari = citesteAdnotarilePentruFiecareImagine(nume_personaj)
    folder_imagini = os.path.join('antrenare', nume_personaj)
    imagini = os.listdir(folder_imagini)
    for imagine in imagini:
        imagine_completa = os.path.join(folder_imagini, imagine)
        img = cv.imread(imagine_completa)
        if imagine in adnotari:
            for anotare in adnotari[imagine]:
                x_min, y_min, x_max, y_max = anotare['coords']
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv.imshow('imagine ' + imagine, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

def creeazaFisierTotalAdnotari():
    foldere_personaje = ['dad', 'mom', 'deedee', 'dexter']
    fisier_destinatie = os.path.join('antrenare', 'toate_adnotarile.txt')
    
    with open(fisier_destinatie, 'w') as f_dest:
        for personaj in foldere_personaje:
            cale_adnotari = os.path.join('antrenare', f'{personaj}_annotations.txt')
            with open(cale_adnotari, 'r') as f_sursa:
                for linie in f_sursa:
                    nume_imagine, x_min, y_min, x_max, y_max, personaj_detectat = linie.strip().split()
                    nume_nou = f"{nume_imagine.split('.')[0]}_{personaj}.jpg"
                    f_dest.write(f"{nume_nou} {x_min} {y_min} {x_max} {y_max} {personaj_detectat}\n")
                    

def creareDatasetCropatFisiere():
    # Deschidem folderul exemple_pozitive_total
    folder_sursa = os.path.join('antrenare', 'exemple_pozitive_total')
    adnotations_file = os.path.join('antrenare', 'toate_adnotarile.txt')
    # Creaza folderul exemple_pozitive_cropate
    os.makedirs(os.path.join('antrenare', 'exemple_pozitive_cropate'), exist_ok=True)
    
    with open(adnotations_file, 'r') as f:
        adnotations = f.readlines()
    for adnotation in adnotations:
        nume_imagine, x_min, y_min, x_max, y_max, personaj_detectat = adnotation.strip().split()
        imagine = cv.imread(os.path.join(folder_sursa, nume_imagine))
        imagine_cropata = imagine[int(y_min):int(y_max), int(x_min):int(x_max)]
        if imagine_cropata.shape[0] < 12 or imagine_cropata.shape[1] < 12:
            print(f"Skipping {nume_imagine} - too small: {imagine_cropata.shape}")
            continue
            
        nume_baza = os.path.splitext(nume_imagine)[0]
        nume_imagine_cropata = f"{nume_baza}_{personaj_detectat}.jpg"
        numar = 1
        while os.path.exists(os.path.join('antrenare', 'exemple_pozitive_cropate', nume_imagine_cropata)):
            nume_imagine_cropata = f"{nume_baza}_{personaj_detectat}_{numar}.jpg"
            numar += 1
        cv.imwrite(os.path.join('antrenare', 'exemple_pozitive_cropate', nume_imagine_cropata), imagine_cropata)

if __name__ == "__main__":
    afiseazaFiecareImagineCuBoundingBoxes("dexter")
    # creeazaFisierTotalAdnotari()

