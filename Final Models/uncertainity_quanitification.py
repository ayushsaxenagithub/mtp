import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load data
file_path = 'combined_data_all_reynolds_20PI_100PI.xlsx'
df = pd.read_excel(file_path).dropna(subset=['Pressure_Strain', 'FileType', 'Lx'])
X = df[['y/delta', 'y^+', 'Production', 'Turbulent_Transport', 'Viscous_Transport', 'Pressure_Transport', 'Viscous_Dissipation', 'Re']]
y = df['Pressure_Strain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define DNN Model as a function
def build_dnn_model():
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Models dictionary
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=100, verbose=0, random_state=42),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
    'DNN': build_dnn_model()
}

# Train and predict with uncertainty estimation
for name, model in models.items():
    if name == 'DNN':
        # Fit the DNN model
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
        # Predict with dropout at inference time (Monte Carlo dropout)
        predictions = np.array([model(X_test_scaled, training=True) for _ in range(100)])
        mean_prediction = np.mean(predictions, axis=0).flatten()
        std_deviation = np.std(predictions, axis=0).flatten()
    else:
        # Fit other models
        model.fit(X_train_scaled, y_train)
        # Bootstrap predictions for tree-based models or single prediction for others
        if hasattr(model, 'estimators_'):
            predictions = np.stack([estimator.predict(X_test_scaled) for estimator in model.estimators_])
            mean_prediction = np.mean(predictions, axis=0)
            std_deviation = np.std(predictions, axis=0)
        else:
            mean_prediction = model.predict(X_test_scaled)
            std_deviation = np.zeros_like(mean_prediction)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.errorbar(y_test, mean_prediction, yerr=std_deviation, fmt='o', alpha=0.5, label=f'{name} Uncertainty')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual Pressure Strain')
    plt.ylabel('Predicted Pressure Strain')
    plt.title(f'Comparison of Model Predictions with Uncertainty for {name}')
    plt.legend()
    plt.show()
