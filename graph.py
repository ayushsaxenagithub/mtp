import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('output.csv')  # Replace with your actual data file

# Define the columns for plotting
columns = ['y/delta', 'Production', 'Turbulent_Transport', 'Viscous_Transport', 'Pressure_Transport', 'Viscous_Dissipation', 'Re', 'Pressure_Strain']

# Creating line plots for each feature against 'y/delta'
for col in columns[1:]:  # Skip 'y/delta' itself
    plt.figure(figsize=(10, 4))
    plt.plot(data['y/delta'], data[col], label=col)
    plt.xlabel('y/delta')
    plt.ylabel(col)
    plt.title(f'Line Plot: {col} vs y/delta')
    plt.legend()
    plt.show()

# Creating histograms for each feature
for col in columns:
    plt.figure(figsize=(8, 4))
    plt.hist(data[col], bins=20, alpha=0.7, label=col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col}')
    plt.legend()
    plt.show()

# Creating boxplots for each feature
for col in columns:
    plt.figure(figsize=(8, 4))
    plt.boxplot(data[col])
    plt.xlabel(col)
    plt.title(f'Boxplot of {col}')
    plt.show()

# Scatter plots between 'Pressure_Strain' and all other features
for col in columns[:-1]:  # Exclude 'Pressure_Strain' itself
    plt.figure(figsize=(8, 6))
    plt.scatter(data[col], data['Pressure_Strain'], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('Pressure_Strain')
    plt.title(f'Scatter Plot: Pressure_Strain vs {col}')
    plt.show()