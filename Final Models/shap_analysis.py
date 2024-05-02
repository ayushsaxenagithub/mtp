# import matplotlib.pyplot as plt
# import joblib
# import pandas as pd
# import numpy as np

# # Load the model
# model = joblib.load('final_model.joblib')

# # Assuming 'X' is loaded or available from previous code, and it contains the correct feature names
# feature_names = ['y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport',
#                  'Pressure_Transport', 'Viscous_Dissipation', 'Re']

# # Get feature importances from the model (assuming the model is a fitted RandomForest)
# importances = model.named_steps['rf'].feature_importances_

# # Create a DataFrame for better visualization and manipulation
# features_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# # Plotting feature importance
# plt.figure(figsize=(10, 6))
# plt.title('Feature Importance')
# plt.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
# plt.xlabel('Importance')
# plt.ylabel('Features')
# plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
# plt.show()

# # Plotting a donut graph of feature importances
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.pie(features_df['Importance'], labels=features_df['Feature'], autopct='%1.1f%%', startangle=90, pctdistance=0.85, colors=plt.cm.Paired(np.arange(len(importances))))

# # Draw a circle at the center to turn the pie into a donut
# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)

# ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Feature Importance Donut Chart')
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load the dataset
file_path = 'combined_data_all_reynolds_20PI_100PI.xlsx'
df = pd.read_excel(file_path).dropna(subset=['Pressure_Strain', 'FileType', 'Lx'])
X = df[['y/delta', 'y^+', 'Production', 'Turbulent_Transport', 'Viscous_Transport', 'Pressure_Transport', 'Viscous_Dissipation', 'Re']]
y = df['Pressure_Strain']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', None)  # Placeholder for the model
])

# Dictionary of models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, min_samples_split=5, min_samples_leaf=1, max_depth=20, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, objective='reg:squarederror', random_state=42),
    'CatBoost': CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, verbose=0, random_state=42),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=42),
}

# Analyze each model
for name, model in models.items():
    pipeline.set_params(model=model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Create the appropriate SHAP explainer
    if name == 'AdaBoost':
        explainer = shap.KernelExplainer(pipeline.named_steps['model'].predict, shap.sample(X_train, 100))  # Using KernelExplainer for AdaBoost
    else:
        explainer = shap.TreeExplainer(pipeline.named_steps['model'])  # Using TreeExplainer for tree-based models

    shap_values = explainer.shap_values(X_test)

    # Calculate the mean absolute SHAP values for each feature and convert to percentage
    shap_sum = np.abs(shap_values).mean(axis=0)
    shap_percentage = (shap_sum / shap_sum.sum()) * 100

    # Donut plot
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(shap_percentage, labels=X.columns, autopct='%1.1f%%', startangle=140, pctdistance=0.85, wedgeprops=dict(width=0.3))
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    plt.title(f'SHAP Value Percentages for {name}')
    plt.show()

    # Summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f'SHAP Summary Plot for {name}')
    plt.show()
