import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score

# Load the model
model = joblib.load('final_model.joblib')

# Load the new Excel data
file_path = 'testing.xlsx'  # Ensure this path is correct
new_data = pd.read_excel(file_path)

# Assuming columns to drop are named 'FileType' and 'Dataset'
# Check if these columns are present and drop them
if 'FileType' in new_data.columns:
    new_data.drop('FileType', axis=1, inplace=True)
if 'Dataset' in new_data.columns:
    new_data.drop('Dataset', axis=1, inplace=True)

# Check for NaN values and handle them
if new_data.isnull().any().any():
    print("Data contains NaN values. Handling them...")
    # Drop rows where target is NaN, assuming 'Pressure_Strain' is the target
    new_data.dropna(subset=['Pressure_Strain'], inplace=True)  
    # Fill NaNs in features with the mean (or choose another method)
    new_data.fillna(new_data.mean(), inplace=True)

# Assuming the dataset includes a target column named 'Pressure_Strain'
X_new = new_data[['y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport',
                  'Pressure_Transport', 'Viscous_Dissipation', 'Re']]
y_actual = new_data['Pressure_Strain']

# Predict using the loaded model
y_pred = model.predict(X_new)

# Calculate R^2 Score (Coefficient of Determination) for accuracy
r2 = r2_score(y_actual, y_pred)
accuracy_percent = r2 * 100
print(f"Accuracy (%): {accuracy_percent:.2f}%")

# Visualization of predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_actual, label='Actual Pressure Strain', color='red', marker='o')
plt.plot(y_pred, label='Predicted Pressure Strain', color='blue', linestyle='--', marker='x')
plt.title('Comparison of Actual vs. Predicted Pressure Strain')
plt.xlabel('Sample Index')
plt.ylabel('Pressure Strain')
plt.legend()
plt.show()

# Residual Plot
residuals = y_actual - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, color='green')
plt.hlines(y=0, xmin=0, xmax=len(residuals), colors='black', linestyles='--')
plt.title('Residuals of Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Residuals')
plt.show()
