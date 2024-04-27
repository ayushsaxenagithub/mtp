import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from joblib import dump

def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    features = df.drop(['Pressure_Strain', 'FileType', 'Dataset','Lx'], axis=1).values
    target = df['Pressure_Strain'].values

    # Reshape data for GRU [samples, time steps, features]
    features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(-1, features.shape[2])).reshape(features.shape)

    # Save the scaler for later use
    dump(scaler, 'scaler.joblib')

    return features, target

def build_model(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        GRU(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    filepath = 'combined_data_all_reynolds_20PI_100PI.xlsx'
    features, target = load_and_preprocess_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    model.save('gru_pressure_strain_model.h5')

if __name__ == '__main__':
    main()
