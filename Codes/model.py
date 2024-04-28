import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data from Excel file
data = pd.read_excel('training.xlsx')
# Assuming 'pressure_strain' is the target variable
X_train = data.drop(['Pressure_Strain', 'FileType', 'Dataset','Lx'], axis=1).values
Y_train = data['Pressure_Strain'].values

# Split data into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Model configuration
input_dim = X_train.shape[1]
output_dim = 1
hidden_units = 20  # Number of neurons in each layer
num_layers = 5    # Total number of layers

# Create a Sequential model
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_dim, activation='relu'))
for _ in range(1, num_layers):
    model.add(Dense(hidden_units, activation='relu'))
model.add(Dense(output_dim, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=10, validation_data=(X_val, Y_val))

# Save the model
model.save('pressure_strain_model.h5')
