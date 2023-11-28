# import numpy as np
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# # Load the saved model
# model = load_model('my_model_with_Re.h5')

# # Assuming you have new input data for prediction in the order:
# # 'y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport', 'Pressure_Transport', 'Viscous_Dissipation', 'Re'
# # Replace this with actual data
# new_data = np.array([[...]])  # Replace with your new data

# # Normalize the new data (assuming the same range as the training data)
# scaler = MinMaxScaler()
# new_data_scaled = scaler.fit_transform(new_data)

# # Make predictions
# predictions = model.predict(new_data_scaled)

# # Output the predictions
# print(predictions)



import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = load_model('my_model_with_Re.h5')

def make_prediction(input_data):
    # Normalize the input data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform([input_data])

    # Make prediction
    prediction = model.predict(input_data_scaled)
    return prediction[0][0]