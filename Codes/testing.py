import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from Excel file
test_data = pd.read_excel('testing.xlsx')
# Drop columns that are not features. Update column names as necessary.
X_test = test_data.drop(['Pressure_Strain', 'FileType', 'Dataset'], axis=1).values
Y_test = test_data['Pressure_Strain'].values

# Load the trained model
model = tf.keras.models.load_model('pressure_strain_model.h5')

# Predict the outputs for the test data
Y_pred = model.predict(X_test).flatten()

# Evaluate the model on the test data
loss = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}')

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# Print some of the actual and predicted values for inspection
print("Some actual values:", Y_test[:10])
print("Some predicted values:", Y_pred[:10])

# Plotting the results with line plots for better trend visualization
plt.figure(figsize=(15, 7))
plt.plot(Y_test, label='Actual', color='blue', marker='o', linestyle='-', markersize=5)
plt.plot(Y_pred, label='Predicted', color='red', marker='x', linestyle='--', markersize=5)
plt.title('Comparison of Actual and Predicted Pressure Strain')
plt.xlabel('Sample Index')
plt.ylabel('Pressure Strain')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and plot errors
errors = Y_test - Y_pred
plt.figure(figsize=(15, 7))
plt.plot(errors, label='Prediction Errors', color='purple', marker='o', linestyle='-')
plt.title('Prediction Errors (Actual - Predicted)')
plt.xlabel('Sample Index')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()
