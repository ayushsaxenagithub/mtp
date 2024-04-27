import pandas as pd

# Load the data
file_path = 'combined_data_all_reynolds_20PI.xlsx'
df = pd.read_excel(file_path)

# Print initial info
print("Initial data info:")
print(df.info())

# Remove rows where any elements are NaN
df = df.dropna(how='any')

# Convert columns to numeric where appropriate, errors='coerce' will convert non-convertible values to NaN
numeric_cols = ['y/delta', 'y^+', 'Production', 'Turbulent_Transport', 'Viscous_Transport', 
                'Pressure_Strain', 'Pressure_Transport', 'Viscous_Dissipation', 'Balance']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove duplicate rows, considering all columns to identify duplicates
df = df.drop_duplicates()

# Print final info after cleaning
print("Data info after cleaning:")
print(df.info())

# Save the cleaned DataFrame back to an Excel file
cleaned_file_path = 'cleaned_combined_data_all_reynolds_20PI.xlsx'
df.to_excel(cleaned_file_path, index=False)
print(f"Cleaned data has been saved to {cleaned_file_path}")
