"""
This script defines the PyTorch Dataset for the humanoid navigation task.
It reads preprocessed CSV files and loads corresponding image sequences.
"""

import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HumanoidNavDataset(Dataset):
    """
    A PyTorch Dataset for loading humanoid navigation data.

    It reads preprocessed CSV files from a directory, where each CSV represents
    a single trajectory. It then loads the corresponding image and state data
    in sequences.
    """

    def __init__(self, processed_data_dir, csv_files, sequence_length=16, use_rgb=False, transform=None, state_mean=None, state_std=None):
        """
        Initializes the dataset.

        Args:
            processed_data_dir (str): Directory containing the processed CSV files.
            csv_files (list): A list of CSV filenames to be included in this dataset.
            sequence_length (int): The length of the sequences to be returned.
            use_rgb (bool): Whether to load and return RGB images.
            transform (callable, optional): A function/transform to apply to the images.
            state_mean (torch.Tensor, optional): The mean of the state vector for normalization.
            state_std (torch.Tensor, optional): The std of the state vector for normalization.
        """
        self.data_dir = processed_data_dir
        self.csv_files = csv_files
        self.sequence_length = sequence_length
        self.use_rgb = use_rgb
        self.transform = transform
        self.state_mean = state_mean
        self.state_std = state_std

        self.trajectories = []
        self.cumulative_sizes = [0]
        self.base_image_path = os.path.join(os.path.dirname(self.data_dir), 'raw')

        if not csv_files:
            print("Warning: No CSV files provided to the dataset.")

        for csv_file in csv_files:
            file_path = os.path.join(self.data_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                if len(df) >= self.sequence_length:
                    self.trajectories.append(df)
                    num_sequences = len(df) - self.sequence_length + 1
                    self.cumulative_sizes.append(self.cumulative_sizes[-1] + num_sequences)
            except Exception as e:
                print(f"Warning: Could not read or process {csv_file}. Error: {e}")
        
        self.total_sequences = self.cumulative_sizes[-1]

        print(f"Found {len(self.trajectories)} valid trajectories.")
        print(f"Total number of possible sequences: {self.total_sequences}")

    def __len__(self):
        """Returns the total number of possible sequences."""
        return self.total_sequences

    def __getitem__(self, idx):
        """
        Retrieves a single training sample (a sequence of data).
        """
        if idx < 0:
            idx += self.total_sequences
        if not 0 <= idx < self.total_sequences:
            raise IndexError("Index out of range")

        traj_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        start_idx = idx - self.cumulative_sizes[traj_idx]
        sequence_df = self.trajectories[traj_idx].iloc[start_idx : start_idx + self.sequence_length]

        depth_imgs = []
        rgb_imgs = []

        for _, row in sequence_df.iterrows():
            try:
                depth_path = os.path.join(self.base_image_path, row['depth_filename'].replace('/', os.sep))
                depth_img = Image.open(depth_path).convert('L')
                depth_imgs.append(depth_img)

                if self.use_rgb:
                    rgb_path = os.path.join(self.base_image_path, row['rgb_filename'].replace('/', os.sep))
                    rgb_img = Image.open(rgb_path).convert('RGB')
                    rgb_imgs.append(rgb_img)
            except FileNotFoundError as e:
                print(f"ERROR: Image not found at {e.filename}. Please check your data paths.")
                # Return None or a dummy sample to avoid crashing the whole training
                return None

        if self.transform:
            depth_imgs = [self.transform(img) for img in depth_imgs]
            if self.use_rgb:
                rgb_imgs = [self.transform(img) for img in rgb_imgs]

        depth_tensor = torch.stack(depth_imgs)
        rgb_tensor = torch.stack(rgb_imgs) if self.use_rgb else torch.empty(0)

        state_cols = ['distance_to_target', 'angle_to_target', 'current_vel_x', 'current_vel_y', 'current_vel_yaw']
        label_cols = ['cmd_vel_x', 'cmd_vel_y', 'cmd_vel_yaw']

        state_data = sequence_df[state_cols].values.astype(np.float32)
        label_data = sequence_df[label_cols].values.astype(np.float32)

        state_tensor = torch.from_numpy(state_data)
        if self.state_mean is not None and self.state_std is not None:
            state_tensor = (state_tensor - self.state_mean) / self.state_std

        return {
            'depth_img': depth_tensor,
            'rgb_img': rgb_tensor,
            'state': state_tensor,
            'label': torch.from_numpy(label_data)
        }

if __name__ == '__main__':
    # This is a simplified test, run train_all_models.py for a full test.
    print("--- Testing HumanoidNavDataset Initialization ---")
