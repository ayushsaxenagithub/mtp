import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load Excel data
file_path = 'combined_data_all_reynolds_20PI_100PI.xlsx'
df = pd.read_excel(file_path)

# Define input features according to your dataset
features = [
    'y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport',
    'Pressure_Transport', 'Viscous_Dissipation', 'Re'
]

# Drop rows where Pressure_Strain is NaN
df = df.dropna(subset=['Pressure_Strain'])

X = df[features]  # predictor variables
y = df['Pressure_Strain']  # response variable

# Handling NaN values in features by imputation
imputer = SimpleImputer(strategy='mean')  # or median, or most_frequent

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Neural Network Regressor with imputation
mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
pipeline = make_pipeline(imputer, mlp)  # creating a pipeline with the imputer and MLPRegressor

# Train the model with the training data
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("R^2 Score: ", r2)

# Visualization of predictions vs actual
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Pressure Strain')
plt.ylabel('Predicted Pressure Strain')
plt.title('Actual vs Predicted Pressure Strain')
plt.show()
