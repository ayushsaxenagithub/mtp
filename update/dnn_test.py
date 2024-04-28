import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk

def load_data(filepath):
    data = pd.read_excel(filepath)
    features = data.drop(['Pressure_Strain', 'Pressure_Transport', 'FileType', 'Dataset'], axis=1)
    target = data['Pressure_Strain']

    # Handling NaNs in features and target
    features.fillna(features.mean(), inplace=True)
    target.fillna(target.mean(), inplace=True)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, target.values  # Ensure target is also a numpy array

def predict(features):
    model = tf.keras.models.load_model('dnn.h5')
    predictions = model.predict(features)
    # Replace NaN predictions if any
    if np.isnan(predictions).any():
        print("NaNs found in predictions, replacing with zeros")
        predictions = np.nan_to_num(predictions)
    return predictions

def evaluate(predictions, actual):
    if np.isnan(predictions).any():
        print("NaNs found in predictions during evaluation")
    if np.isnan(actual).any():
        print("NaNs found in actual values during evaluation")
    mse = mean_squared_error(actual, predictions)
    return mse

def display_results(actual, predictions, mse):
    root = tk.Tk()
    root.title("Model Predictions and Accuracy")
    tree = ttk.Treeview(root, columns=('Actual', 'Predicted'), show='headings')
    tree.heading('Actual', text='Actual Pressure Strain')
    tree.heading('Predicted', text='Predicted Pressure Strain')
    tree.pack(fill=tk.BOTH, expand=True)
    for act, pred in zip(actual, predictions.flatten()):
        tree.insert('', 'end', values=(act, pred))
    mse_label = ttk.Label(root, text=f"Mean Squared Error: {mse:.4f}", font=("Arial", 16))
    mse_label.pack(pady=10)
    root.mainloop()

def main():
    data_path = 'testing.xlsx'
    features, actual = load_data(data_path)
    predictions = predict(features)
    mse = evaluate(predictions, actual)
    display_results(actual, predictions, mse)

if __name__ == '__main__':
    main()
