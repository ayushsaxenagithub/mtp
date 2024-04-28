import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from joblib import load

# Define paths to models and scaler
model_paths = {
    'dnn': 'dnn.h5',
    'cnn': 'cnn_pressure_strain_model.h5',
    'gru': 'gru_pressure_strain_model.h5',
    'rnn': 'lstm_pressure_strain_model.h5',
    'rf': 'gradient_boosting_model.joblib'  
}

scaler_path = 'scaler.joblib'

# Function to load data and apply transformations
def load_data(filepath, scaler_path):
    data = pd.read_excel(filepath)
    features = data.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1)
    
    # Load and apply scaler
    scaler = load(scaler_path)
    features = scaler.transform(features)

    # Reshape for GRU: Assuming each feature is treated as a separate timestep with 1 feature per timestep
    features = np.reshape(features, (features.shape[0], features.shape[1], 1))
    
    return features, data['Pressure_Strain']


# Function to predict using all models
def predict_all(features):
    predictions = {}
    for model_name, path in model_paths.items():
        if 'rf' in model_name:
            # For sklearn models
            from sklearn.externals import joblib
            model = joblib.load(path)
            pred = model.predict(features)  # RF does not need 3D data
        else:
            # For Keras models, reshape if necessary
            if 'cnn' in model_name or 'gru' in model_name:
                features_reshaped = np.expand_dims(features, axis=2)
                model = tf.keras.models.load_model(path)
                pred = model.predict(features_reshaped).flatten()
            else:
                model = tf.keras.models.load_model(path)
                pred = model.predict(features).flatten()
        predictions[model_name] = pred
    return predictions

# Function to combine predictions
def combine_predictions(predictions):
    # Simple averaging method
    combined = np.mean(np.array(list(predictions.values())), axis=0)
    return combined

# Function to calculate accuracy
def calculate_accuracy(actual, predicted):
    accuracy = np.mean(np.abs((actual - predicted) / actual)) * 100
    return 100 - accuracy  # Mean Absolute Percentage Error

# Main execution function
def main():
    filepath = 'testing.xlsx'
    scaler_path = 'scaler.joblib'  # Ensure this is the correct path to your scaler file
    features, actual = load_data(filepath, scaler_path)  # Now calling with both required arguments
    predictions = predict_all(features)
    combined_predictions = combine_predictions(predictions)
    accuracy = calculate_accuracy(actual, combined_predictions)
    
    print(f"Combined Model Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
