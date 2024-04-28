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

    # Reshape data for LSTM and scale
    features = np.reshape(features.values, (features.shape[0], 1, features.shape[1]))
    features = scaler.transform(features.reshape(-1, features.shape[2])).reshape(features.shape)
    
    return features, target

def predict(model_path, features):
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(features)
    return predictions.flatten()

def calculate_accuracy(actual, predictions):
    correct_predictions = np.sum(np.round(predictions) == actual)
    total_predictions = len(actual)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy    

def display_results(actual, predictions, accuracy):
    root = tk.Tk()
    root.title("LSTM Model Predictions and Accuracy")
    tree = ttk.Treeview(root, columns=('Actual', 'Predicted'), show='headings')
    tree.heading('Actual', text='Actual Pressure Strain')
    tree.heading('Predicted', text='Predicted Pressure Strain')
    tree.pack(fill=tk.BOTH, expand=True)

    for act, pred in zip(actual, predictions):
        tree.insert('', 'end', values=(act, pred))

    # Display accuracy
    accuracy_label = ttk.Label(root, text=f"Accuracy: {accuracy:.2f}%")
    accuracy_label.pack(pady=10)

    root.mainloop()

def main():
    data_path = 'testing.xlsx'
    features, actual = load_data(data_path)
    predictions = predict('lstm_pressure_strain_model.h5', features)
    accuracy = calculate_accuracy(actual, predictions)
    display_results(actual, predictions, accuracy)

if __name__ == '__main__':
    main()
