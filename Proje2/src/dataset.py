import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(224, 224)):
    """
    Loads images from the specified directory. 
    Assumes standard 'yes' (tumor) and 'no' (no tumor) subdirectories.
    """
    images = []
    labels = []
    
    classes = {'no': 0, 'yes': 1}
    
    for cls, val in classes.items():
        folder_path = os.path.join(data_dir, cls)
        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} does not exist. Skipping.")
            continue
            
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(val)
                
    images = np.array(images, dtype='float32') / 255.0  # Normalize
    labels = np.array(labels, dtype='int32')
    
    return images, labels

def get_train_val_data(data_dir, img_size=(224, 224), test_size=0.2, random_state=42):
    images, labels = load_data(data_dir, img_size)
    if len(images) == 0:
        return None, None, None, None
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val
