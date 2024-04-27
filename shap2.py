import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('output.csv')  # Replace with your actual data file

# Define features and target variable
features = data[['y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport', 'Pressure_Transport', 'Viscous_Dissipation', 'Re']]
target = data['Pressure_Strain']

# Normalize the data
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.2, random_state=42)

# Load the pre-trained ANN model
model = load_model('my_model_with_Re.h5')

# SHAP Kernel Explainer
explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test)

# Calculate total absolute SHAP values for each instance
total_shap_values = np.absolute(shap_values).sum(axis=1)

# Calculate percentage contribution of each parameter
shap_percent_contributions = (np.absolute(shap_values) / total_shap_values[:, None]) * 100

# Take the mean along the rows
mean_shap_percent = shap_percent_contributions.mean(axis=0)

# Flatten the array to make it 1D
mean_shap_percent = mean_shap_percent.flatten()

# Plot donut chart
labels = ['y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport', 'Pressure_Transport', 'Viscous_Dissipation', 'Re']
plt.pie(mean_shap_percent, labels=labels, autopct='%1.1f%%', startangle=90)
plt.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
plt.axis('equal')
plt.title('Average Percent Contribution of Parameters')
plt.show()
