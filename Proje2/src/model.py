from sklearn.neural_network import MLPClassifier
import joblib

def build_model():
    # Use MLPClassifier for a neural network that works well on native python without TensorFlow
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42, verbose=True)
    return model

def save_model(model, path):
    joblib.dump(model, path)
    
def load_saved_model(path):
    return joblib.load(path)
