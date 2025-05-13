'''
Parses through all the 2024 data and creates start time (of day) histograms for each month.
Use this to analyze changes in daily bike traffic patterns from month-to-month
'''
import pandas as pd
import zipfile
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import os

# Show all rows
pd.set_option('display.max_rows', None)

# Show all columns
pd.set_option('display.max_columns', None)

# Prevent wide columns from being truncated
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=np.inf)


N_STATIONS = 2799 # via https://en.wikipedia.org/wiki/Citi_Bike

# ALL ZIPFILE FILES
ZIP_FILES = [#'202401-citibike-tripdata.csv.zip', '202402-citibike-tripdata.csv.zip', '202403-citibike-tripdata.csv.zip', 
             #'202404-citibike-tripdata.csv.zip', '202405-citibike-tripdata.zip', '202406-citibike-tripdata.zip', 
             #'202407-citibike-tripdata.zip', '202408-citibike-tripdata.zip', '202410-citibike-tripdata.zip',
             '202411-citibike-tripdata.zip', '202412-citibike-tripdata.zip']

output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# global id_map
# global data

def read_citibike_zip(zip_path):
    print("Reading zipfile" + zip_path)
    file_name = zip_path.replace('.csv', '').replace('.zip', '').replace('data/', '')
    print("Filename: " + file_name)

    df = None

    with zipfile.ZipFile(zip_path, 'r') as z:
        print(z.namelist())
        # csv_name = z.namelist()[0]  # assumes one CSV per zip
        # with z.open(csv_name) as f:
        #     df = pd.read_csv(f)
        csv_files = [name for name in z.namelist() if name.endswith('.csv')]
        dfs = []

        for csv_name in csv_files:
            print("csv_name:", csv_name)
            with z.open(csv_name) as f:
                try:
                    df_part = pd.read_csv(f)
                    dfs.append(df_part)
                except:
                    print(f"Error. Skipping file: {csv_name}")

        # Combine all into a single DataFrame
        df = pd.concat(dfs, ignore_index=True)


    df = df.rename(columns={
        'start_station_id': 'start_id',
        'start_lat': 'start_lat',
        'start_lng': 'start_lng',
        'end_station_id': 'end_id',
        'end_lat': 'end_lat',
        'end_lng': 'end_lng'
    })

    print("df.columns:", df.columns)
    print("df.shape:", df.shape)

    # 1. REMOVE NULL VALUES
    df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 2. CONVERT ALL START_IDS, END_IDS TO STRINGS (SO THAT UNIQUE_IDS COUNTS SMTH LIKE 6501.23 AND '6501.23' AS THE SAME VALUE)
    # WANNA MAP STATION IDS TO SOME NICE INTEGER VALUE 0 TO N_STATIONS - 1
    df['start_id'] = df['start_id'].astype(str)
    df['end_id'] = df['end_id'].astype(str)

    unique_ids = pd.unique(pd.concat([df['start_id'], df['end_id']]))
    id_map = {id_: i for i, id_ in enumerate(unique_ids)}
    print("unique_ids.shape", unique_ids.shape)
    df['start_id'] = df['start_id'].map(id_map)
    df['end_id'] = df['end_id'].map(id_map)
    
    # 2.5. SAMPLE 20 TO PRINT TO DOUBLE CHECK DATA LOOKS OK
    sample_df = df.sample(n=20, random_state=42)
    print("\nRandom Sample of 20 Rows:")
    print(sample_df[['start_station_name', 'start_id', 'end_station_name', 'end_id']])

    # 3. Plot histogram to determine what hours of the day have the most amount of bike trips
    # Categories: 1-2, 2-3, ..., 12-13, 13-14, ..., 21-22, 23-00
    # 3. Plot histogram to determine what hours of the day have the most bike trips
    df['started_at'] = pd.to_datetime(df['started_at'])
    # dtype: datetime64[ns] – print(df['started_at'].dtype)
    df['start_hour'] = df['started_at'].dt.hour

    plt.figure(figsize=(12, 6))
    plt.hist(df['start_hour'], bins=24, range=(0, 24), edgecolor='black', align='left')
    plt.xticks(range(24))
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Trips')
    plt.title('Number of Bike Trips by Hour of Day')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(output_dir,  f'{file_name}.png')
    plt.savefig(plot_path)

    return df

if __name__ == "__main__":
    # A is your adjacency matrix (N x N)
    # stations maps matrix row indices to station_id and coordinates
    for zip_file in ZIP_FILES:
        zip_path = "data/" + zip_file
        df = read_citibike_zip(zip_path)