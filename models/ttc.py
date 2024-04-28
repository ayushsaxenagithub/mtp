import pandas as pd

# Absolute paths to the Excel files
file_path1 = r'C:\Users\ayush\OneDrive\Desktop\code\MTP\mtp\models\testing.xlsx'
file_path2 = r'C:\Users\ayush\OneDrive\Desktop\code\MTP\mtp\models\training.xlsx'  # Corrected the filename if it was a typo

# Load the data from each file
data1 = pd.read_excel(file_path1)
data2 = pd.read_excel(file_path2)

# Combine the dataframes
combined_data = pd.concat([data1, data2], ignore_index=True)

# Save the combined dataframe to a new Excel file
combined_data.to_excel(r'C:\Users\ayush\OneDrive\Desktop\code\MTP\mtp\models\combined_training.xlsx', index=False)
