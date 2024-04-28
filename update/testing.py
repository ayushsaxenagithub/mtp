import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

def load_and_preprocess_test_data(filepath, scaler_path):
    df = pd.read_excel(filepath)
    features = df.drop(['Pressure_Strain', 'Pressure_Transport', 'FileType', 'Dataset'], axis=1)
    target = df['Pressure_Strain']

    # Load the scaler used in training to transform test data
    scaler = pd.read_pickle(scaler_path)
    features = scaler.transform(features)
    
    return features, target

def predict_and_evaluate(model_path, X_test, y_test):
    # Load the saved model
    model = load_model(model_path)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - y_test))
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((predictions - y_test) ** 2)
    # Calculate R-squared
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return mae, mse, r_squared

def main():
    test_filepath = 'testing.xlsx'
    model_path = 'final_model.keras'
    scaler_path = 'scaler.pkl'
    
    X_test, y_test = load_and_preprocess_test_data(test_filepath, scaler_path)
    mae, mse, r_squared = predict_and_evaluate(model_path, X_test, y_test)
    
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"R-squared: {r_squared:.3f}")

if __name__ == "__main__":
    main()
