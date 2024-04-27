import requests
import pandas as pd

# Base URL
base_url = "https://turbulence.oden.utexas.edu/couette2018/data/"

# Dictionary of Reynolds numbers and their corresponding file types and names
reynolds_numbers = {
    220: "LM_Couette_R0220_100PI_RSTE",
    220: "LM_Couette_R0220_020PI_RSTE"
}

def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def process_data(data, name, re_number, file_type):
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
        df['Re'] = re_number
        df['FileType'] = file_type  # Identify as 'prof' or 'stdev'
        return df
    else:
        return None

master_df = pd.DataFrame()

# Process each Reynolds number
for re_number, prefix in reynolds_numbers.items():
    file_types = ['prof', 'stdev']  # Types of files
    for file_type in file_types:
        filenames = [f"{prefix}_uu_{file_type}.dat", f"{prefix}_vv_{file_type}.dat",
                     f"{prefix}_ww_{file_type}.dat", f"{prefix}_uv_{file_type}.dat",
                     f"{prefix}_k_{file_type}.dat"]
        
        for filename in filenames:
            url = base_url + filename
            data = download_data(url)
            if data:
                dataset_name = filename.replace(f'{prefix}_', '').replace(f'_{file_type}.dat', '')
                df = process_data(data, dataset_name, re_number, file_type)
                if df is not None:
                    master_df = pd.concat([master_df, df], ignore_index=True)
                    print(f"Data from {url} ({file_type}) has been added to the master DataFrame.")
                else:
                    print(f"Failed to process data from {url}")
            else:
                print(f"Failed to download data from {url}")

# Save the combined DataFrame to an Excel file
master_df.to_excel('combined_data_all_reynolds_20PI.xlsx', index=False, float_format="%.18f")
print("All data has been saved in a single sheet 'combined_data_all_reynolds_20PI.xlsx'")
