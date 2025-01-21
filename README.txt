Pentru a rula proiectul, avem nevoie de:

- Python 3.10
- OpenCV (cv2)
- NumPy
- PyTorch
- Torchvision 
- scikit-image
- scikit-learn
- Pillow
- matplotlib
- tqdm
- ultralytics (YOLO)

(Optionale)
- CUDA Toolkit
- cuDNN


Proiectul se ruleaza in aceasta structura, deci in folderul CAVA-2024-TEMA2.
Vor fi pregenerati descriptorii (deja se afla toate in acest folder), modelele, inclusiv SGD-ul si CNN-ul, YOLO-ul, si fisierele de antrenare.
Toate imaginile pozitive-negative, pe care am facut antrenarea, plus toate preprocesarile necesare sunt deja in structura proiectului.

Pentru a rula sliding window-ul direct, incarcand modelul si tot:

1. Se pun imaginile care trebuie validate in folderul validare/validare/ (ca si in exemplul initial)
2. Se pun in folderul validare/ fisierele text pentru ground truth:
    - task1_gt_validare.txt
    - task2_dad_gt_validare.txt
    - task2_mom_gt_validare.txt
    - task2_deedee_gt_validare.txt
    - task2_dexter_gt_validare.txt

3. Daca sunt prezente, se sterg folderele:
    - antrenare/fisiere_salvate_algoritm/detections/
    - antrenare/fisiere_salvate_algoritm/task1
    - antrenare/fisiere_salvate_algoritm/task2
(Acestea sunt foldere generate in urma rularii sliding window-ului in modelul nostru)

4. Daca sunt prezente, se sterg folderele:
    - code/yolo_model/detections
(Acesta contine fisiere generate in urma rularii modelului YOLO)

5. Ne ducem cu terminalul (cd) in folderul code/
6. Rulam comanda:
    python RunProject.py
7. Introducem 2 pentru a rula doar sliding window-ul plus CNN-ul (acum asteptam sa ruleze)

8. Rulam comanda:
    python ParseTextIntoNumpy.py
9. Introducem 3 (optiunea "both")

Acum avem toate fisiere numpy generate pentru toate task-urile cu modelul construit de mine.

10. (Rulam partea de YOLO)
    - Ne ducem cu terminalul (cd) in folderul code/yolo_model/
    - Rulam comanda:
        python Detect.py

Acum avem toate fisiere numpy generate pentru toate task-urile cu modelul YOLO.

10. Ne ducem cu terminalul (cd) in folderul evaluare/cod_evaluare/
11. Schimbam in evalueaza_solutie.py calea (comentam / decomentam) liniile 173-174 in functie de ce model vrem sa evaluam
    
    # solution_path_root = "../../antrenare/fisiere_salvate_algoritm/"
    solution_path_root = "../../code/yolo_model/detections/"
    (Pentru evaluarea modelului YOLO)

    solution_path_root = "../../antrenare/fisiere_salvate_algoritm/"
    # solution_path_root = "../../code/yolo_model/detections/"
    (Pentru evaluarea modelului meu)

12. Rulam comanda:
    python evalueaza_solutie.py

Vor aparea toate preciziile pentru toate task-urile.

