import tkinter as tk
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler



# from predict import make_prediction  # Importing the prediction function

# def predict():
#     # Collect the input data
#     input_data = [
#         float(entry_y_delta.get()),
#         float(entry_production.get()),
#         float(entry_turbulent_transport.get()),
#         float(entry_viscous_transport.get()),
#         float(entry_pressure_transport.get()),
#         float(entry_viscous_dissipation.get()),
#         float(entry_re.get())
#     ]

#     # Get prediction from predict.py
#     prediction = make_prediction(input_data)
    
#     # Display the prediction
#     label_result.config(text=f"Predicted Pressure Strain: {prediction}")







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
    model = load_model('my_model_with_Re.h5')

    # Normalize the input data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform([input_data])

    # Predict
    prediction = model.predict(input_data_scaled)
    
    # Display the prediction
    label_result.config(text=f"Predicted Pressure Strain: {prediction[0][0]}")

# Set up the GUI
root = tk.Tk()
root.title("Pressure Strain Predictor")

# Create and place input fields
tk.Label(root, text="y/delta").grid(row=0, column=0)
entry_y_delta = tk.Entry(root)
entry_y_delta.grid(row=0, column=1)

tk.Label(root, text="Production").grid(row=1, column=0)
entry_production = tk.Entry(root)
entry_production.grid(row=1, column=1)

tk.Label(root, text="Turbulent Transport").grid(row=2, column=0)
entry_turbulent_transport = tk.Entry(root)
entry_turbulent_transport.grid(row=2, column=1)

tk.Label(root, text="Viscous Transport").grid(row=3, column=0)
entry_viscous_transport = tk.Entry(root)
entry_viscous_transport.grid(row=3, column=1)

tk.Label(root, text="Pressure Transport").grid(row=4, column=0)
entry_pressure_transport = tk.Entry(root)
entry_pressure_transport.grid(row=4, column=1)

tk.Label(root, text="Viscous Dissipation").grid(row=5, column=0)
entry_viscous_dissipation = tk.Entry(root)
entry_viscous_dissipation.grid(row=5, column=1)

tk.Label(root, text="Reynolds Number (Re)").grid(row=6, column=0)
entry_re = tk.Entry(root)
entry_re.grid(row=6, column=1)

# Prediction button
button_predict = tk.Button(root, text="Predict", command=predict)
button_predict.grid(row=7, column=0, columnspan=2)

# Result label
label_result = tk.Label(root, text="Predicted Pressure Strain: ")
label_result.grid(row=8, column=0, columnspan=2)

root.mainloop()






