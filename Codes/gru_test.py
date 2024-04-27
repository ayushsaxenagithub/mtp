import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk
from joblib import load

def load_data(filepath):
    data = pd.read_excel(filepath)
    features = data.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1)
    target = data['Pressure_Strain']

    # Load the scaler that was fitted during training
    scaler = load('scaler.joblib')

    # Reshape data for GRU and scale
    features = np.reshape(features.values, (features.shape[0], 1, features.shape[1]))
    features = scaler.transform(features.reshape(-1, features.shape[2])).reshape(features.shape)
    
    return features, target

def predict(model_path, features):
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(features)
    return predictions.flatten()

def calculate_accuracy(actual, predictions, tolerance=0.1):
    # Define accuracy as the percentage of predictions within 'tolerance' of actual values
    return np.mean(np.abs(predictions - actual) <= tolerance * np.abs(actual)) * 100

def display_results(actual, predictions):
    accuracy = calculate_accuracy(actual, predictions)
    root = tk.Tk()
    root.title("GRU Model Predictions and Accuracy")
    tree = ttk.Treeview(root, columns=('Actual', 'Predicted'), show='headings')
    tree.heading('Actual', text='Actual Pressure Strain')
    tree.heading('Predicted', text='Predicted Pressure Strain')
    tree.pack(fill=tk.BOTH, expand=True)
    for act, pred in zip(actual, predictions):
        tree.insert('', 'end', values=(act, pred))
    
    # Display accuracy
    accuracy_label = ttk.Label(root, text=f"Accuracy: {accuracy:.2f}% within tolerance")
    accuracy_label.pack(pady=10)

    root.mainloop()

def main():
    data_path = 'testing.xlsx'
    features, actual = load_data(data_path)
    predictions = predict('gru_pressure_strain_model.h5', features)
    display_results(actual, predictions)

if __name__ == '__main__':
    main()
