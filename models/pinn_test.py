import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from joblib import load
import tkinter as tk
from tkinter import ttk

class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

def load_data(filepath, scaler_path):
    data = pd.read_excel(filepath)
    features = data.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1)
    target = data['Pressure_Strain']

    scaler = load(scaler_path)
    features = scaler.transform(features)
    
    return features, target

def predict(model, features):
    predictions = model.predict(features)
    return predictions.flatten()

def display_results(actual, predictions):
    root = tk.Tk()
    root.title("PINN Model Predictions")
    tree = ttk.Treeview(root, columns=('Actual', 'Predicted'), show='headings')
    tree.heading('Actual', text='Actual Pressure Strain')
    tree.heading('Predicted', text='Predicted Pressure Strain')
    tree.pack(fill=tk.BOTH, expand=True)

    for act, pred in zip(actual, predictions):
        tree.insert('', 'end', values=(act, pred))

    root.mainloop()

def main():
    data_path = 'testing.xlsx'
    scaler_path = 'scaler.joblib'

    features, actual = load_data(data_path, scaler_path)
    
    model = PINN()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.load_weights('pinn_pressure_strain_model.h5')  # Ensure this matches the saved model weights

    predictions = predict(model, features)
    display_results(actual, predictions)

if __name__ == '__main__':
    main()
