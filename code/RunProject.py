from Parameters import *
from FacialDetectorDexter import *
import numpy as np
import os
import glob
from RunModelTrainer import train_model
from RunClassifierOnData import run_classifier

def main():
    print("Choose operation:")
    print("1. Train model")
    print("2. Run classifier on data")
    print("3. Both")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        train_model()
    elif choice == "2":
        run_classifier()
    elif choice == "3":
        facial_detector = train_model()
        run_classifier()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()