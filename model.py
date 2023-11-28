import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

# Build the ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(features_normalized.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, validation_split=0.2, epochs=100)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Model Loss: {loss}")

# Save the model
model.save('my_model_with_Re.h5')