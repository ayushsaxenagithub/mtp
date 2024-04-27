import requests
import pandas as pd


import requests
import pandas as pd

# Base URL
base_url = "https://turbulence.oden.utexas.edu/couette2018/data/"

# List of filenames to be downloaded
filenames = [
    "LM_Couette_R0220_100PI_RSTE_uu_prof.dat",
    "LM_Couette_R0220_100PI_RSTE_uu_stdev.dat",
    "LM_Couette_R0220_100PI_RSTE_vv_prof.dat",
    "LM_Couette_R0220_100PI_RSTE_vv_stdev.dat",
    "LM_Couette_R0220_100PI_RSTE_ww_prof.dat",
    "LM_Couette_R0220_100PI_RSTE_ww_stdev.dat",
    "LM_Couette_R0220_100PI_RSTE_uv_prof.dat",
    "LM_Couette_R0220_100PI_RSTE_uv_stdev.dat",
    "LM_Couette_R0220_100PI_RSTE_k_prof.dat",
    "LM_Couette_R0220_100PI_RSTE_k_stdev.dat"
]

def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def process_data(data, name):
    lines = data.split('\n')
    columns = None
    data_list = []

    for line in lines:
        if line.startswith('%') and 'y/delta' in line:
            columns = line.lstrip('%').strip().split()
            continue
        if not line.startswith('%') and columns is not None:
            data_list.append(line.split())

    if columns:
        df = pd.DataFrame(data_list, columns=columns)
        df = df.apply(pd.to_numeric, errors='coerce')
        # Add a column to identify the dataset
        df['Dataset'] = name
        return df
    else:
        return None

# Initialize a master DataFrame to hold all concatenated data
master_df = pd.DataFrame()

for filename in filenames:
    url = base_url + filename
    data = download_data(url)
    if data:
        # Extract a name for the dataset based on the filename
        dataset_name = filename.replace('LM_Couette_R0093_100PI_RSTE_', '').replace('.dat', '')
        df = process_data(data, dataset_name)
        if df is not None:
            # Append the individual DataFrame to the master DataFrame
            master_df = pd.concat([master_df, df], ignore_index=True)
            print(f"Data from {url} has been added to the master DataFrame.")
        else:
            print(f"Failed to process data from {url}")
    else:
        print(f"Failed to download data from {url}")

# Save the master DataFrame to a single Excel sheet
master_df.to_excel('combined_single_sheet_data.xlsx', index=False, float_format="%.18f")
print("All data has been saved in a single sheet 'combined_single_sheet_data.xlsx'")
