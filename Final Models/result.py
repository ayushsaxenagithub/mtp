import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load Excel data
file_path = 'combined_data_all_reynolds_20PI_100PI.xlsx'
df = pd.read_excel(file_path).dropna(subset=['Pressure_Strain'])
X = df[['y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport',
        'Pressure_Transport', 'Viscous_Dissipation', 'Re']]
y = df['Pressure_Strain']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline setup without Polynomial Features for quicker execution
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning setup
param_grid = {
    'rf__n_estimators': [100],  # Reduced number for speed
    'rf__max_depth': [10, 20],  # Fewer options
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

# Randomized Search with fewer iterations and less cross-validation
random_search = RandomizedSearchCV(pipeline, param_grid, n_iter=5, cv=3, scoring='r2', random_state=42)
random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)

# Evaluate the best model found from the RandomizedSearch
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100  # Percentage accuracy from R^2 score
print("Test MSE:", mse)
print("Test RMSE:", rmse)
print("Test R^2:", r2)
print("Accuracy (%):", accuracy)

# Visualization of predictions vs actual
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Pressure Strain')
plt.ylabel('Predicted Pressure Strain')
plt.title('Actual vs Predicted Pressure Strain')
plt.savefig('prediction_accuracy_plot.png')  # Save the plot as a PNG file
plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame({
    'Actual Pressure Strain': y_test,
    'Predicted Pressure Strain': y_pred
})
results_df.to_csv('model_results.csv', index=False)
