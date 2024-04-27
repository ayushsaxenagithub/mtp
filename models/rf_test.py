import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from joblib import load
from sklearn.metrics import mean_squared_error

def load_data(filepath, scaler_file):
    data = pd.read_excel(filepath)
    features = data.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1)
    target = data['Pressure_Strain']

    # Handle NaNs in the test data
    features.fillna(features.mean(), inplace=True)
    target.fillna(target.mean(), inplace=True)

    scaler = load(scaler_file)
    features = scaler.transform(features)
    return features, target

def predict(model_path, features):
    model = load(model_path)
    return model.predict(features)

def calculate_accuracy(actual, predictions):
    mse = mean_squared_error(actual, predictions)
    accuracy = 100 - np.sqrt(mse)  # Example custom accuracy metric
    return mse, accuracy

def display_results(actual, predictions, mse, accuracy):
    root = tk.Tk()
    root.title("Gradient Boosting Model Predictions and Accuracy")
    tree = ttk.Treeview(root, columns=('Actual', 'Predicted'), show='headings')
    tree.heading('Actual', text='Actual Pressure Strain')
    tree.heading('Predicted', text='Predicted Pressure Strain')
    tree.pack(fill=tk.BOTH, expand=True)
    for act, pred in zip(actual, predictions):
        tree.insert('', 'end', values=(act, pred))

    accuracy_label = ttk.Label(root, text=f"Accuracy: {accuracy:.2f}%")
    accuracy_label.pack(pady=10)
    mse_label = ttk.Label(root, text=f"Mean Squared Error: {mse:.4f}")
    mse_label.pack(pady=10)
    root.mainloop()

def main():
    data_path = 'testing.xlsx'
    features, actual = load_data(data_path, 'scaler_gb.joblib')
    predictions = predict('gradient_boosting_model.joblib', features)
    mse, accuracy = calculate_accuracy(actual, predictions)
    display_results(actual, predictions, mse, accuracy)

if __name__ == '__main__':
    main()
