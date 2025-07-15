"""
Data preprocessing script for the humanoid navigation task.

This script takes raw CSV data containing robot pose and image filenames,
and processes it to generate features and labels suitable for training
an end-to-end model.

Processing steps include:
1. Optional filtering of pose data (Moving Average or Butterworth).
2. Calculation of velocity labels (vel_x, vel_y, vel_yaw).
3. Calculation of egocentric target features (distance_to_target, angle_to_target).
4. Saving the processed data into a new CSV file.
"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def apply_filter(df, filter_type='butterworth', window_size=5, butter_order=3, butter_cutoff=0.1):
    """
    Applies a filter to the pose data (x, y, yaw) in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'x', 'y', 'yaw' columns.
        filter_type (str): 'moving_average', 'butterworth', or None.
        window_size (int): Window size for the moving average filter.
        butter_order (int): Order of the Butterworth filter.
        butter_cutoff (float): Cutoff frequency for the Butterworth filter.

    Returns:
        pd.DataFrame: DataFrame with filtered pose data.
    """
    if filter_type is None:
        print("No filter applied.")
        return df

    df_filtered = df.copy()
    cols_to_filter = ['x', 'y', 'yaw']

    if filter_type == 'moving_average':
        print(f"Applying moving average filter with window size {window_size}...")
        for col in cols_to_filter:
            df_filtered[col] = df[col].rolling(window=window_size, min_periods=1, center=True).mean()

    elif filter_type == 'butterworth':
        print(f"Applying Butterworth filter with order={butter_order}, cutoff={butter_cutoff}...")
        b, a = butter(butter_order, butter_cutoff, btype='low', analog=False)
        for col in cols_to_filter:
            # Apply zero-phase filter
            df_filtered[col] = filtfilt(b, a, df[col])

    else:
        print(f"Warning: Unknown filter type '{filter_type}'. No filter applied.")

    return df_filtered

def preprocess_file(input_path, output_path, filter_type, **filter_kwargs):
    """
    Preprocesses a single raw data CSV file.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed CSV file.
        filter_type (str): The type of filter to apply.
        **filter_kwargs: Keyword arguments for the filter function.
    """
    print(f"--- Processing file: {os.path.basename(input_path)} ---")

    # 1. Load data
    df = pd.read_csv(input_path)

    # 2. Apply optional filtering
    df_processed = apply_filter(df, filter_type=filter_type, **filter_kwargs)

    # 3. Define the target as the last position in the trajectory
    target_x = df_processed['x'].iloc[-1]
    target_y = df_processed['y'].iloc[-1]
    print(f"Target position set to: ({target_x:.2f}, {target_y:.2f})")

    # 4. Calculate egocentric target features for each timestep
    # These will be the inputs for the MLP_head
    dx = target_x - df_processed['x']
    dy = target_y - df_processed['y']
    df_processed['distance_to_target'] = np.sqrt(dx**2 + dy**2)

    world_angle_to_target = np.arctan2(dy, dx)
    angle_diff = world_angle_to_target - df_processed['yaw']
    df_processed['angle_to_target'] = angle_diff.apply(normalize_angle)

    # 5. Calculate velocity labels (ground truth for the model)
    # Assuming 10Hz, so dt = 0.1s
    dt = 0.1
    df_processed['vel_x'] = df_processed['x'].diff() / dt
    df_processed['vel_y'] = df_processed['y'].diff() / dt

    # Handle yaw wrapping for angular velocity calculation
    yaw_diff = df_processed['yaw'].diff().apply(normalize_angle)
    df_processed['vel_yaw'] = yaw_diff / dt

    # The first row will have NaN for velocities, back-fill it
    df_processed.fillna(method='bfill', inplace=True)

    # 6. Select final columns and save
    # These are the only columns the Dataset class will need
    final_columns = [
        'depth_filename',
        'rgb_filename',
        'distance_to_target',
        'angle_to_target',
        'vel_x',
        'vel_y',
        'vel_yaw'
    ]
    # Ensure rgb_filename column exists, if not, create it with empty values
    if 'rgb_filename' not in df_processed.columns:
        df_processed['rgb_filename'] = ''

    df_final = df_processed[final_columns]
    df_final.to_csv(output_path, index=False)
    print(f"Successfully saved processed data to: {os.path.basename(output_path)}\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw humanoid navigation data.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing raw CSV files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save processed CSV files.")
    parser.add_argument('--filter', type=str, default='butterworth', choices=['butterworth', 'moving_average', 'none'],
                        help="Type of filter to apply to pose data. 'none' to disable.")
    parser.add_argument('--window', type=int, default=5, help="Window size for moving average filter.")
    parser.add_argument('--order', type=int, default=3, help="Order for Butterworth filter.")
    parser.add_argument('--cutoff', type=float, default=0.1, help="Cutoff frequency for Butterworth filter.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    filter_type = None if args.filter == 'none' else args.filter
    filter_kwargs = {
        'window_size': args.window,
        'butter_order': args.order,
        'butter_cutoff': args.cutoff
    }

    for filename in os.listdir(args.input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            preprocess_file(input_path, output_path, filter_type, **filter_kwargs)

if __name__ == '__main__':
    main()

