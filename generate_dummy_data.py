
"""
Generates a dummy dataset for testing the humanoid navigation task.

This script creates:
1. A sample CSV file (`sample_trajectory.csv`) simulating a robot's trajectory.
2. Corresponding dummy RGB and Depth image files in respective subdirectories.
"""

import os
import time
import pandas as pd
import numpy as np
from PIL import Image

def create_dummy_csv(output_dir, filename, num_points=50, hz=10):
    """
    Creates a dummy CSV file simulating a robot's trajectory.

    Args:
        output_dir (str): The directory to save the CSV file.
        num_points (int): The number of data points to generate.
        hz (int): The sampling frequency.

    Returns:
        str: The path to the created CSV file.
    """
    print(f"--- Generating dummy CSV: {filename} ---")
    
    data = []
    current_time = int(time.time() * 1000) # Millisecond timestamp
    x, y, yaw = np.random.rand(3) * np.array([2, 2, np.pi]) # Random start pose
    
    # Simulate a simple, slightly varying velocity profile
    base_vel_x = 0.4 + np.random.rand() * 0.2  # 0.4 to 0.6 m/s
    base_vel_yaw = (np.random.rand() - 0.5) * 0.4 # -0.2 to 0.2 rad/s
    dt = 1.0 / hz

    # Store previous pose to calculate velocity
    prev_x, prev_y, prev_yaw = x, y, yaw

    for i in range(num_points):
        timestamp = current_time + int(i * dt * 1000)
        
        # Create filenames based on timestamp
        depth_filename = f"depth/{timestamp}.png"
        rgb_filename = f"rgb/{timestamp}.png"

        # Calculate current velocity based on pose change
        # This simulates the velocity that would be in /odom
        current_vel_x = (x - prev_x) / dt
        current_vel_y = (y - prev_y) / dt
        current_vel_yaw = (yaw - prev_yaw) / dt

        # The "label" is the command that *caused* this state change.
        # For dummy data, we'll assume the command was the velocity we aimed for.
        cmd_vel_x = base_vel_x + np.sin(i * 0.1) * 0.1 # Add some variation
        cmd_vel_y = 0.0 # Assume no sideways command
        cmd_vel_yaw = base_vel_yaw

        data.append({
            'depth_filename': depth_filename,
            'rgb_filename': rgb_filename,
            'x': x,
            'y': y,
            'yaw': yaw,
            'current_vel_x': current_vel_x,
            'current_vel_y': current_vel_y,
            'current_vel_yaw': current_vel_yaw,
            'cmd_vel_x': cmd_vel_x,
            'cmd_vel_y': cmd_vel_y,
            'cmd_vel_yaw': cmd_vel_yaw
        })

        # Update previous pose for next iteration's velocity calculation
        prev_x, prev_y, prev_yaw = x, y, yaw

        # Update pose for the next step based on the "commanded" velocity
        x += cmd_vel_x * np.cos(yaw) * dt
        y += cmd_vel_x * np.sin(yaw) * dt
        yaw += cmd_vel_yaw * dt

    df = pd.DataFrame(data)
    
    # Define target as the last point in the trajectory
    target_x, target_y = df['x'].iloc[-1], df['y'].iloc[-1]
    
    # Calculate distance and angle to target for each point
    dx = target_x - df['x']
    dy = target_y - df['y']
    df['distance_to_target'] = np.sqrt(dx**2 + dy**2)
    
    # Angle of the target relative to the robot's current orientation
    angle_to_target_global = np.arctan2(dy, dx)
    df['angle_to_target'] = angle_to_target_global - df['yaw']
    # Normalize angle to be within [-pi, pi]
    df['angle_to_target'] = np.arctan2(np.sin(df['angle_to_target']), np.cos(df['angle_to_target']))

    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    
    print(f"Successfully created dummy CSV at: {csv_path}")
    return csv_path

def create_dummy_images(csv_path, base_dir):
    """
    Creates dummy image files based on filenames in a CSV.

    Args:
        csv_path (str): Path to the CSV file containing filenames.
        base_dir (str): The base directory where 'rgb' and 'depth' folders will be.
    """
    print(f"--- Generating dummy images for {os.path.basename(csv_path)} ---")
    df = pd.read_csv(csv_path)

    # Create dummy images
    for _, row in df.iterrows():
        # RGB Image (3 x 480 x 640)
        rgb_path = os.path.join(base_dir, row['rgb_filename'])
        os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
        rgb_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        rgb_img.save(rgb_path)

        # Depth Image (1 x 480 x 640)
        depth_path = os.path.join(base_dir, row['depth_filename'])
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        # Use 16-bit integer for depth, common for depth cameras
        depth_img = Image.fromarray(np.zeros((480, 640), dtype=np.uint16))
        depth_img.save(depth_path)

    print(f"Successfully created {len(df)} dummy RGB and Depth images.")

def main():
    NUM_TRAJECTORIES = 5
    # Define the directory for our raw dummy data
    raw_data_dir = 'data/raw'
    os.makedirs(raw_data_dir, exist_ok=True)


    for i in range(NUM_TRAJECTORIES):
        filename = f"trajectory_{i+1}.csv"
        
        # Step 1: Create the CSV file for one trajectory
        csv_file_path = create_dummy_csv(raw_data_dir, filename)
        # Step 2: Create the corresponding image files for that trajectory
        create_dummy_images(csv_file_path, raw_data_dir)
        print("-" * 20)

    print(f"\nDummy dataset generation complete! Created {NUM_TRAJECTORIES} trajectories.")
    print(f"The CSVs now contain the required 5D state and 3D command labels.")

if __name__ == '__main__':
    main()
