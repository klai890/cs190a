import pandas as pd
import zipfile
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from datetime import timedelta
import os

'''
def split_dataset(X_interval, y_interval, df, id_map):
    """
    Splits trip data into training and testing windows with temporal consistency.

    Parameters:
    - X_interval (timedelta): Time length for training (e.g., timedelta(days=7)).
    - y_interval (timedelta): Time length for testing (e.g., timedelta(days=1)).
    - df (pd.DataFrame): Trip data. Must include 'start_time' and 'start_id'.
    - id_map (dict): Maps station IDs to integer indices [0, N-1].
    - time_granularity (str): Pandas frequency string (e.g., 'H' for hourly, 'D' for daily).

    Returns:
    - X: List of training data, a time series of [(Dt, Ft), (Dt+1, Ft+1), ... ]
    - y: List of testing data, a time series of [(Dt, Ft), (Dt+1, Ft+1), ... ]
    """

    # Sort by time
    df = df.sort_values('start_time')
    df = df[df['start_id'].isin(id_map)]  # Ensure all stations are known

    # Time range
    start_time = df['start_time'].min()
    end_time = df['start_time'].max()

    X = []
    y = []
    current_time = start_time

    while current_time + X_interval + y_interval <= end_time:
        X_start = current_time
        X_end = current_time + X_interval
        y_end = X_end + y_interval

        # Filter trips
        X_df = df[(df['start_time'] >= X_start) & (df['start_time'] < X_end)]
        y_df = df[(df['start_time'] >= X_end) & (df['start_time'] < y_end)]


        # Initialize matrices: [num_stations, num_time_steps]
        num_stations = len(id_map)
        demand_X = np.zeros((num_stations, num_stations))
        demand_y = np.zeros((num_stations, num_stations))

        # Create X demand matrix
        X_counts = X_df.groupby(['start_id', 'end_id']).size()
        for (i, j), count in X_counts.items():
            demand_X[i][j] = count

        # Todo: Create X feature matrix
        X.append((demand_X))

        # Create y demand matrix
        y_counts = y_df.groupby(['start_id', 'end_id']).size()
        for (i, j), count in y_counts.items():
            demand_y[i][j] = count

        # Todo: Create y feature matrix
        y.append((demand_y))

        # Slide window
        current_time += (X_interval + y_interval)

    return X, y
'''

def split_dataset(X_interval, y_interval, df, id_map):
    """
    Splits trip data into training (X) and testing (y) sequences for GNN+RNN model input.
    
    Parameters:
    - X_interval (timedelta): Length of training window (e.g., timedelta(days=7))
    - y_interval (timedelta): Length of testing window (e.g., timedelta(days=1))
    - df (pd.DataFrame): Trip data with columns ['start_time', 'start_id', 'end_id']
    - id_map (dict): Mapping from station IDs to indices [0, N-1]
    
    Returns:
    - X: list of np.arrays of shape (seq_len_X, num_stations, 2) for outgoing & incoming demand
    - y: list of np.arrays of shape (seq_len_y, num_stations, 2) similarly for testing
    """
    # Sort data by start_time
    df = df.sort_values('started_at')
    # Commented out this line because it's incredibly buggy and filters out everything.
    # df = df[df['start_id'].isin(id_map) & df['end_id'].isin(id_map)]  # filter known stations
    
    start_time = df['started_at'].min()
    end_time = df['ended_at'].max()

    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    
    num_stations = len(id_map)
    X, y = [], []
    current_time = start_time

    # Calculate number of discrete time steps in X and y intervals (e.g., days)
    # We'll assume time granularity = 1 day; adjust if needed
    time_granularity = timedelta(hours=4)
    seq_len_X = X_interval // time_granularity
    seq_len_y = y_interval // time_granularity
    
    while current_time + X_interval + y_interval <= end_time:
        # Build training sequence
        X_seq = []
        for step in range(seq_len_X):
            t_start = current_time + step * time_granularity
            t_end = t_start + time_granularity            
            print(f"t_start: {t_start}")
            print(f"t_end: {t_end}")
            window_df = df[(df['started_at'] >= t_start) & (df['started_at'] < t_end)]
            
            demand_matrix = np.zeros((num_stations, num_stations))
            for _, row in window_df.iterrows():
                i = row['start_id']
                j = row['end_id']
                demand_matrix[i, j] += 1
            
            outgoing = demand_matrix.sum(axis=1)
            incoming = demand_matrix.sum(axis=0)
            node_features = np.stack([outgoing, incoming], axis=1)  # shape (num_stations, 2)
            X_seq.append(node_features)
            # print("X_seq:", X_seq)
        X.append(np.stack(X_seq, axis=0))  # shape (seq_len_X, num_stations, 2)
        
        # Build testing sequence
        y_seq = []
        y_start_time = current_time + X_interval
        for step in range(seq_len_y):
            t_start = y_start_time + step * time_granularity
            t_end = t_start + time_granularity
            window_df = df[(df['started_at'] >= t_start) & (df['started_at'] < t_end)]
            
            demand_matrix = np.zeros((num_stations, num_stations))
            for _, row in window_df.iterrows():
                i = row['start_id']
                j = row['end_id']
                demand_matrix[i, j] += 1
            
            outgoing = demand_matrix.sum(axis=1)
            incoming = demand_matrix.sum(axis=0)
            node_features = np.stack([outgoing, incoming], axis=1)
            y_seq.append(node_features)
            # print("y_seq:", y_seq)
        y.append(np.stack(y_seq, axis=0))  # shape (seq_len_y, num_stations, 2)
        
        # Move to next window
        current_time += (X_interval + y_interval)
    
    return X, y
