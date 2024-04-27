import requests
import pandas as pd

# Base URL
base_url = "https://turbulence.oden.utexas.edu/couette2018/data/"

# List of filenames for Re 500 with 100 PI
filenames = [
    "LM_Couette_R0500_100PI_RSTE_uu_prof.dat",
    "LM_Couette_R0500_100PI_RSTE_uu_stdev.dat",
    "LM_Couette_R0500_100PI_RSTE_vv_prof.dat",
    "LM_Couette_R0500_100PI_RSTE_vv_stdev.dat",
    "LM_Couette_R0500_100PI_RSTE_ww_prof.dat",
    "LM_Couette_R0500_100PI_RSTE_ww_stdev.dat",
    "LM_Couette_R0500_100PI_RSTE_uv_prof.dat",
    "LM_Couette_R0500_100PI_RSTE_uv_stdev.dat",
    "LM_Couette_R0500_100PI_RSTE_k_prof.dat",
    "LM_Couette_R0500_100PI_RSTE_k_stdev.dat"
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
        df['Dataset'] = name
        df['Re'] = 500
        return df
    else:
        return None

master_df = pd.DataFrame()

for filename in filenames:
    url = base_url + filename
    data = download_data(url)
    if data:
        dataset_name = filename.replace('LM_Couette_R0500_100PI_RSTE_', '').replace('.dat', '')
        df = process_data(data, dataset_name)
        if df is not None:
            master_df = pd.concat([master_df, df], ignore_index=True)
            print(f"Data from {url} has been added to the master DataFrame.")
        else:
            print(f"Failed to process data from {url}")
    else:
        print(f"Failed to download data from {url}")

# Correct filename to include the .xlsx extension
# Save the DataFrame to an Excel file with the correct filename including the .xlsx extension
master_df.to_excel('Re_500_combined_single_sheet_data.xlsx', index=False, float_format="%.18f")
print("All data has been saved in a single sheet 'Re_500_combined_single_sheet_data.xlsx'")

