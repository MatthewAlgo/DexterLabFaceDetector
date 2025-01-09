import numpy as np
from FacialDetectorDexter import FacialDetectorDexter
from Parameters import Parameters

def evaluate_results_task1(solution_path, ground_truth_path, verbose=0):
    # Încarcă detectiile + scorurile + numele de imagini
    detections = np.load(solution_path + "detections_all_faces.npy",
                        allow_pickle=True, fix_imports=True, encoding='latin1')
    print(detections.shape)

    scores = np.load(solution_path + "scores_all_faces.npy",
                    allow_pickle=True, fix_imports=True, encoding='latin1')
    print(scores.shape)
    
    file_names = np.load(solution_path + "file_names_all_faces.npy",
                        allow_pickle=True, fix_imports=True, encoding='latin1')
    print(file_names.shape)

    # Creăm o instanță temporară pentru evaluare
    params = Parameters()
    detector = FacialDetectorDexter(params)
    detector.eval_detections(detections, scores, file_names)

if __name__ == "__main__":
    solution_path = "./antrenare/fisiere_salvate_algoritm/task1/"
    ground_truth_path = "./validare/task1_gt_validare.txt"
    evaluate_results_task1(solution_path, ground_truth_path)
