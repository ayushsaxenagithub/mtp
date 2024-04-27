import pandas as pd

# Load the datasets
df_20PI = pd.read_excel('20PI.xlsx')
df_100PI = pd.read_excel('100PI.xlsx')

# Add a new column to each DataFrame to denote the Lx value
df_20PI['Lx'] = 20
df_100PI['Lx'] = 100

# Combine the two DataFrames
combined_df = pd.concat([df_20PI, df_100PI], ignore_index=True)

# Save the combined DataFrame to a new Excel file
combined_df.to_excel('combined_data_all_reynolds_20PI_100PI.xlsx', index=False)

print("Combined Excel file has been saved as 'combined_data_all_reynolds_20PI_100PI.xlsx'")
