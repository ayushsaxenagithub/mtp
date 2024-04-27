import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk
import numpy as np

def load_data(filepath):
    data = pd.read_excel(filepath)
    # Exclude non-feature columns explicitly and check for NaNs
    features = data.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1)
    target = data['Pressure_Strain']

    # Check and handle NaNs
    if features.isna().any().any() or target.isna().any():
        features = features.fillna(features.mean())  # Filling NaNs with the mean of each column
        target = target.fillna(target.mean())  # Filling NaNs in the target

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, target

def predict(model_path, features):
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(features)
    return predictions

def evaluate(predictions, actual):
    mse = mean_squared_error(actual, predictions)
    return mse

def calculate_accuracy(actual, predictions, tolerance=0.1):
    # Define accuracy as the percentage of predictions within 'tolerance' of actual values
    correct_predictions = np.sum(np.abs(predictions.flatten() - actual) <= tolerance * np.abs(actual))
    total_predictions = len(actual)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

def display_results(actual, predictions, mse, accuracy):
    root = tk.Tk()
    root.title("Model Predictions and Accuracy")
    tree = ttk.Treeview(root, columns=('Actual', 'Predicted'), show='headings')
    tree.heading('Actual', text='Actual Pressure Strain')
    tree.heading('Predicted', text='Predicted Pressure Strain')
    tree.pack(fill=tk.BOTH, expand=True)
    for act, pred in zip(actual, predictions.flatten()):
        tree.insert('', 'end', values=(act, pred))

    # Display MSE and accuracy
    mse_label = ttk.Label(root, text=f"Mean Squared Error: {mse:.4f}", font=("Arial", 16))
    mse_label.pack(pady=10)
    accuracy_label = ttk.Label(root, text=f"Accuracy: {accuracy:.2f}%", font=("Arial", 16))
    accuracy_label.pack(pady=10)

    root.mainloop()

def main():
    data_path = 'testing.xlsx'
    model_path = 'dnn.h5'
    features, actual = load_data(data_path)
    predictions = predict(model_path, features)
    mse = evaluate(predictions, actual)
    accuracy = calculate_accuracy(actual, predictions)
    display_results(actual, predictions, mse, accuracy)

if __name__ == '__main__':
    main()
