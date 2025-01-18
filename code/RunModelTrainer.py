from Parameters import *
from FacialDetectorDexter import *
import numpy as np
import os
import glob

def train_model():
    params = Parameters()
    facial_detector = FacialDetectorDexter(params)
    
    try:
        print("\nCollecting training data...")
        training_examples_dict = {}
        train_labels_dict = {}
        
        for window_size in params.sizes_array:
            print(f"\nProcessing window size {window_size}...")
            
            try:
                positive_features = facial_detector.get_positive_descriptors(window_size)
                negative_features = facial_detector.get_negative_descriptors(window_size)
                
                if len(positive_features) == 0 or len(negative_features) == 0:
                    print(f"Warning: No features extracted for window size {window_size}")
                    continue
                    
                training_examples_dict[window_size] = np.vstack((positive_features, negative_features))
                train_labels_dict[window_size] = np.hstack((np.ones(len(positive_features)), 
                                                          np.zeros(len(negative_features))))
                                                          
            except Exception as e:
                print(f"Error processing window size {window_size}: {e}")
                continue
        
        if not training_examples_dict:
            raise RuntimeError("No valid training data collected")
            
        # Train classifier
        facial_detector.train_classifier(training_examples_dict, train_labels_dict)
        
        return facial_detector
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    train_model()
