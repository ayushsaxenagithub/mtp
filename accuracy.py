import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import MeanSquaredError

# Load your data
data = pd.read_csv('output.csv')  # Replace with your actual data file

# Include Reynolds number in features
features = data[['y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport', 'Pressure_Transport', 'Viscous_Dissipation', 'Re']]
target = data['Pressure_Strain']

# Normalize the data
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.2, random_state=42)

# Load the original model
original_model = load_model('my_model_with_Re.h5')

# Compile the model with the appropriate metrics
original_model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])

# Evaluate the original model
original_loss = original_model.evaluate(X_test, y_test)
print(f"Original Model Loss: {original_loss}")
