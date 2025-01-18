from Parameters import *
from RunClassifierOnData import run_detector, run_cnn_classifier
from RunModelTrainer import train_model  # Changed import to correct file

def main():
    print("\nChoose operation:")
    print("1. Train classifier")
    print("2. Run face detection")
    print("3. Run CNN classification on detections")
    print("4. Run complete pipeline (detection + classification)")
    
    choice = input("Enter your choice (1-4): ")
    
    params = Parameters()
    
    if choice == "1":
        print("\nTraining classifier...")
        train_model()
    elif choice == "2":
        print("\nRunning face detection...")
        run_detector(params)
    elif choice == "3":
        print("\nRunning CNN classification on existing detections...")
        run_cnn_classifier(params)
    elif choice == "4":
        print("\nRunning complete pipeline...")
        run_detector(params)
        run_cnn_classifier(params)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()