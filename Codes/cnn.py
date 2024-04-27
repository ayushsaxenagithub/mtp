import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def load_data(filepath):
    # Load dataset
    df = pd.read_excel(filepath)
    features = df.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1)
    target = df['Pressure_Strain']

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Reshape for Conv1D (batch, steps, channels)
    features = np.expand_dims(features, axis=2)
    
    return features, target

def build_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

def main():
    # Load and prepare data
    filepath = 'combined_data_all_reynolds_20PI_100PI.xlsx'  # Path to your data file
    features, target = load_data(filepath)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Build and train model
    model = build_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save('cnn_pressure_strain_model.h5')

if __name__ == '__main__':
    main()
