import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk
from joblib import load

def load_data(filepath, scaler_path):
    data = pd.read_excel(filepath)
    features = data.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1)
    
    # Load and apply scaler
    scaler = load(scaler_path)
    features = scaler.transform(features)
    features = np.expand_dims(features, axis=2)  # Reshape for Conv1D

    target = data['Pressure_Strain']
    return features, target

def predict(model_path, features):
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(features)
    return predictions.flatten()

def calculate_accuracy(actual, predictions):
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    return 100 - mape  # Convert to accuracy

def display_results(actual, predictions):
    root = tk.Tk()
    root.title("CNN Model Predictions")
    tree = ttk.Treeview(root, columns=('Actual', 'Predicted'), show='headings')
    tree.heading('Actual', text='Actual Pressure Strain')
    tree.heading('Predicted', text='Predicted Pressure Strain')
    tree.pack(fill=tk.BOTH, expand=True)

    # Calculate accuracy
    accuracy = calculate_accuracy(actual, predictions)

    for act, pred in zip(actual, predictions):
        tree.insert('', 'end', values=(act, pred))

    # Display accuracy
    accuracy_label = ttk.Label(root, text=f"Accuracy: {accuracy:.2f}%", font=("Arial", 16))
    accuracy_label.pack(pady=10)

    root.mainloop()

def main():
    data_path = 'testing.xlsx'
    scaler_path = 'scaler.joblib'
    features, actual = load_data(data_path, scaler_path)
    
    model_path = 'cnn_pressure_strain_model.h5'
    predictions = predict(model_path, features)
    display_results(actual, predictions)

if __name__ == '__main__':
    main()
