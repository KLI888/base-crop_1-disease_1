from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.pkl'
try:
    model = joblib.load(MODEL_PATH)
except:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not found. Please train the model first.'}), 500

    try:
        data = request.json
        features = [
            float(data.get('N')),
            float(data.get('P')),
            float(data.get('K')),
            float(data.get('temperature')),
            float(data.get('humidity')),
            float(data.get('ph')),
            float(data.get('rainfall'))
        ]
        
        prediction = model.predict([features])
        recommended_crop = prediction[0]
        
        return jsonify({'recommendation': recommended_crop})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
