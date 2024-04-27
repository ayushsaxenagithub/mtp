import pandas as pd

# Load the Excel file
df = pd.read_excel('combined_single_sheet_data.xlsx')

# Drop rows where all elements are NaN
df = df.dropna(how='all')

# Save the DataFrame back to an Excel file with the proper .xlsx extension
df.to_excel('Re_93.xlsx', index=False)
print("Filtered Excel file saved without empty rows.")
