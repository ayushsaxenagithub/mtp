import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('final_model.joblib')

# Assuming 'X' is loaded or available from previous code, and it contains the correct feature names
feature_names = ['y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport',
                 'Pressure_Transport', 'Viscous_Dissipation', 'Re']

# Get feature importances from the model (assuming the model is a fitted RandomForest)
importances = model.named_steps['rf'].feature_importances_

# Create a DataFrame for better visualization and manipulation
features_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
plt.show()

# Plotting a donut graph of feature importances
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(features_df['Importance'], labels=features_df['Feature'], autopct='%1.1f%%', startangle=90, pctdistance=0.85, colors=plt.cm.Paired(np.arange(len(importances))))

# Draw a circle at the center to turn the pie into a donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Feature Importance Donut Chart')
plt.show()
