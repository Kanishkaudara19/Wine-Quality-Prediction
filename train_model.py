import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the wine quality dataset
print("Loading dataset...")
data = pd.read_csv('data/train.csv')

# Display basic information
print("\nDataset Information:")
print(f"Shape: {data.shape}")
print("\nColumns:", data.columns.tolist())
print("\nMissing Values:")
print(data.isnull().sum())

# Separate features and target
X = data.drop(columns=['id', 'quality'])
y = data['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
# Dictionary to store model results
model_results = {}
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train multiple regression models
print("\nTraining models...")
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_results[name] = {'MSE': mse, 'R2': r2}
    trained_models[name] = model

    # Save the model
    with open(f'models/{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Display performance comparison
print("\nModel Performance Comparison:")
for model, metrics in model_results.items():
    print(f"{model}: MSE = {metrics['MSE']:.4f}, R2 = {metrics['R2']:.4f}")

# Find the best model based on MSE
best_model = min(model_results.items(), key=lambda x: x[1]['MSE'])
print("\nBest Model:")
print(f"\nBest Model: {best_model[0]} (MSE = {best_model[1]['MSE']:.4f}, R2 = {best_model[1]['R2']:.4f})")

# Load test data
data_eval = pd.read_csv("data/test.csv")
X_eval = data_eval.drop(columns=['id'])
X_eval_scaled = scaler.transform(X_eval)

# Create a DataFrame to store model predictions
predictions_df = pd.DataFrame({'id': data_eval['id']})

# Predict with each model
for model_name, model in trained_models.items():
    try:
        predictions_df[model_name] = model.predict(X_eval_scaled)  # Store full array
    except Exception as e:
        predictions_df[model_name] = None  # Fill errors with NaN

# Take row-wise mean across all models (ignoring NaN values)
predictions_df['quality'] = predictions_df.iloc[:, 1:].mean(axis=1).round(3)  # Round to nearest integer

# Generate submission file
predictions_df[['id', 'quality']].to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' generated successfully.")

# Save feature names for the web app
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\nTraining complete. All models saved to 'models/' directory.")