import pickle
import os
import numpy as np
from flask import Flask, render_template, request
from setuptools import Extension

app = Flask(__name__)

# Load models
models = {}
model_files = {
    'Linear Regression': 'linear_regression.pkl',
    'Ridge Regression': 'ridge_regression.pkl',
    'Random Forest': 'random_forest.pkl',
    'Gradient Boosting': 'gradient_boosting.pkl'
}

# Try to load all models
for model_name, file_name in model_files.items():
    model_path = os.path.join('models', file_name)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)
    else:
        print(f"Warning: Model file '{model_path}' not found.")

# Load the scaler
try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Warning: Scaler file not found.")
    scaler = None

# Load feature names
try:
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    print("Warning: Feature names file not found.")
    feature_names = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    all_predictions = {}
    
    if request.method == 'POST':
        # Get input values from form
        input_data = []
        for feature in feature_names:
            value = request.form.get(feature.replace(' ', '_'), 0)
            try:
                input_data.append(float(value))
            except ValueError:
                input_data.append(0.0)
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input
        if scaler is not None:
            scaled_input = scaler.transform(input_array)
        else:
            scaled_input = input_array
            
        # Make predictions with each model
        for model_name, model in models.items():
            try:
                pred = model.predict(scaled_input)[0]
                all_predictions[model_name] = round(pred, 2)
            except Exception as e:
                all_predictions[model_name] = f"Error: {str(e)}"
        
        # Aggregate all predictions and get a mean prediction value
        prediction = sum(all_predictions.values()) / len(all_predictions)
    
    return render_template('index.html', 
                           prediction=prediction,
                           all_predictions=all_predictions,
                           feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)