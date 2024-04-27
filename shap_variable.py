import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import shap

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

# Check and reshape shap_values if necessary
if len(shap_values.shape) == 3:
    shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])

# Plot summary plot
shap.summary_plot(shap_values, X_test, feature_names=['y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport', 'Pressure_Transport', 'Viscous_Dissipation', 'Re'])
