# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(filepath):
    # Load the data from an Excel file
    df = pd.read_excel(filepath)

    # Drop non-feature columns and NaN values
    features = df.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1)
    target = df['Pressure_Strain']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def build_and_compile_model(input_shape):
    # Define the model architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

def main():
    # Specify the path to your Excel file
    filepath = 'combined_data_all_reynolds_20PI_100PI.xlsx'

    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Build and compile the DNN model
    model = build_and_compile_model(X_train.shape[1])

    # Train the model
    model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

    # Evaluate the model on the test data
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

    # Save the model
    model.save('dnn.h5')
    print('Model saved to dnn.h5')

if __name__ == "__main__":
    main()
