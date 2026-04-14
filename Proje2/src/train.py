import os
from dataset import get_train_val_data
from model import build_model, save_model

def main():
    data_dir = os.path.join('..', 'data')
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Ensure you place your data in ../data.")
        return
        
    X_train, X_val, y_train, y_val = get_train_val_data(data_dir, img_size=(64, 64))
    
    if X_train is None or len(X_train) == 0:
        print("No images found to train on. Check the data directory.")
        return

    # Flatten the images to 1D vectors for scikit-learn
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    print(f"Training on {len(X_train)} images, validating on {len(X_val)} images.")

    model = build_model()
    
    # Train the model
    model.fit(X_train_flat, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_flat, y_train)
    val_acc = model.score(X_val_flat, y_val)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    
    model_dir = os.path.join('..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'tumor_model.pkl')
    
    save_model(model, model_path)
    print(f"Training complete. Model saved to {model_path}.")

if __name__ == "__main__":
    main()
