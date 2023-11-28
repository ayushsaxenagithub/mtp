import tkinter as tk
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def predict():
    # Collect the input data
    input_data = [
        float(entry_y_delta.get()),
        float(entry_production.get()),
        float(entry_turbulent_transport.get()),
        float(entry_viscous_transport.get()),
        float(entry_pressure_transport.get()),
        float(entry_viscous_dissipation.get()),
        float(entry_re.get())
    ]

    # Load the model
    model = load_model('my_model.keras')

    # Normalize the input data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform([input_data])

    # Predict
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    label_result.config(text=f"Predicted Pressure Strain: {prediction[0][0]}")
    # Assuming error is calculated elsewhere and passed here, replace 'error_value' with the actual error
    # label_error.config(text=f"Prediction Error: {error_value}")

# Set up the GUI
root = tk.Tk()
root.title("Pressure Strain Predictor")
root.configure(bg='lightgray')

# Create and place input fields
# Optional: Add label_error if you plan to display error
# label_error = tk.Label(root, text="Prediction Error: ", bg='lightblue')
# label_error.grid(row=9, column=0, columnspan=2)

root.mainloop()