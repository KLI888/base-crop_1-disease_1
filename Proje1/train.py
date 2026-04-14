import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import urllib.request
import os

DATASET_URL = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
DATASET_PATH = "crop_recommendation.csv"
MODEL_PATH = "model.pkl"

def train():
    print("Fetching dataset...")
    if not os.path.exists(DATASET_PATH):
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        print("Dataset downloaded.")
    else:
        print("Dataset already exists.")
        
    df = pd.read_csv(DATASET_PATH)
    
    print("Preprocessing data...")
    # The dataset has the following columns: N, P, K, temperature, humidity, ph, rainfall, label
    features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully.")

if __name__ == '__main__':
    train()
