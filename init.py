import pandas as pd
import zipfile
import numpy as np
from geopy.distance import geodesic

def read_citibike_zip(zip_path):
    print("Reading zipfile")
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_name = z.namelist()[0]  # assumes one CSV per zip
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    return df

def build_station_adjacency(df):
    print("Building adjacency matrix")
    df = df.rename(columns={
        'start_station_id': 'start_id',
        'start_lat': 'start_lat',
        'start_lng': 'start_lng',
        'end_station_id': 'end_id',
        'end_lat': 'end_lat',
        'end_lng': 'end_lng'
    })

    print("df.columns:", df.columns)
    for col in df.columns:
        print("Unique values in column", col)
        print(np.unique(df[col].astype(str)))

    print("Done")
    start_stations = df[['start_id', 'start_station_name', 'start_lat', 'start_lng']].rename(
        columns={'start_id': 'station_id'})
    end_stations = df[['end_id', 'end_lat', 'end_lng']].rename(
        columns={'end_id': 'station_id', 'end_lat': 'start_lat', 'end_lng': 'start_lng'})

    stations = pd.concat([start_stations, end_stations]).dropna().drop_duplicates('station_id').reset_index(drop=True)
    print("Stations:", stations.size)
    print("Stations:", stations[:10])

    N = len(stations)
    A = np.zeros((N, N))

    A = (A + A.T) / 2  # make symmetric
    return A, stations


if __name__ == "__main__":
    # A is your adjacency matrix (N x N)
    # stations maps matrix row indices to station_id and coordinates
    print("Started main function")
    zip_path = "data/202401-citibike-tripdata.csv.zip"
    df = read_citibike_zip(zip_path)
    A, stations = build_station_adjacency(df)