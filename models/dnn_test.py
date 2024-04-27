import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk

def load_data(filepath):
    # Load and preprocess the data
    data = pd.read_excel(filepath)
    # Exclude non-feature columns explicitly
    features = data.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1)
    target = data['Pressure_Strain']

    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, target

def predict(model_path, features):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(features)
    return predictions

def evaluate(predictions, actual):
    mse = mean_squared_error(actual, predictions)
    return mse

def display_results(actual, predictions, mse):
    # Setup the root GUI window
    root = tk.Tk()
    root.title("Model Predictions and Accuracy")

    # Setup the treeview for displaying data
    tree = ttk.Treeview(root, columns=('Actual', 'Predicted'), show='headings')
    tree.heading('Actual', text='Actual Pressure Strain')
    tree.heading('Predicted', text='Predicted Pressure Strain')
    tree.pack(fill=tk.BOTH, expand=True)

    # Insert data into the treeview
    for act, pred in zip(actual, predictions.flatten()):
        tree.insert('', 'end', values=(act, pred))

    # Display MSE
    mse_label = ttk.Label(root, text=f"Mean Squared Error: {mse:.4f}", font=("Arial", 16))
    mse_label.pack(pady=10)

    # Start the GUI event loop
    root.mainloop()

def main():
    # Paths to your model and data file
    model_path = 'pressure_strain_prediction_model.h5'
    data_path = 'testing.xlsx'

    # Load and predict
    features, actual = load_data(data_path)
    predictions = predict(model_path, features)

    # Evaluate and display
    mse = evaluate(predictions, actual)
    display_results(actual, predictions, mse)

if __name__ == '__main__':
    main()
