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

    def __init__(self, processed_data_dir, csv_files, sequence_length=16, use_rgb=False, transform=None):
        """
        Initializes the dataset.

        Args:
            processed_data_dir (str): Directory containing the processed CSV files.
            csv_files (list): A list of CSV filenames to be included in this dataset.
            sequence_length (int): The length of the sequences to be returned.
            use_rgb (bool): Whether to load and return RGB images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data_dir = processed_data_dir
        self.csv_files = csv_files
        self.sequence_length = sequence_length
        self.use_rgb = use_rgb
        self.transform = transform

        self.trajectories = []
        self.cumulative_sizes = [0]
        self.base_image_path = os.path.join(os.path.dirname(self.data_dir), 'raw')

        # Load all trajectory dataframes and calculate cumulative sizes for indexing
        if not csv_files:
            print("Warning: No CSV files provided to the dataset.")

        for csv_file in csv_files:
            file_path = os.path.join(self.data_dir, csv_file)
            df = pd.read_csv(file_path)
            
            # A trajectory must be at least as long as the sequence length
            if len(df) >= self.sequence_length:
                self.trajectories.append(df)
                # Number of possible start points for a sequence in this trajectory
                num_sequences = len(df) - self.sequence_length + 1
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + num_sequences)
        
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

        # Find which trajectory this index belongs to
        traj_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        
        # Find the start index within that trajectory
        start_idx = idx - self.cumulative_sizes[traj_idx]

        # Get the sequence slice from the dataframe
        sequence_df = self.trajectories[traj_idx].iloc[start_idx : start_idx + self.sequence_length]

        # --- Load Images ---
        depth_imgs = []
        rgb_imgs = []

        for _, row in sequence_df.iterrows():
            # Load depth image (as grayscale)
            # Construct a cross-platform safe path
            depth_path = os.path.join(self.base_image_path, row['depth_filename'].replace('/', os.sep))
            depth_img = Image.open(depth_path).convert('L') # L for grayscale
            depth_imgs.append(depth_img)

            # Load RGB image if requested
            if self.use_rgb:
                # Construct a cross-platform safe path
                rgb_path = os.path.join(self.base_image_path, row['rgb_filename'].replace('/', os.sep))
                rgb_img = Image.open(rgb_path).convert('RGB')
                rgb_imgs.append(rgb_img)

        # Apply transformations and stack
        if self.transform:
            depth_imgs = [self.transform(img) for img in depth_imgs]
            if self.use_rgb:
                rgb_imgs = [self.transform(img) for img in rgb_imgs]

        depth_tensor = torch.stack(depth_imgs)
        rgb_tensor = torch.stack(rgb_imgs) if self.use_rgb else torch.empty(0)

        # --- Get State and Label Data ---
        state_cols = ['distance_to_target', 'angle_to_target']
        label_cols = ['vel_x', 'vel_y', 'vel_yaw']

        state_data = sequence_df[state_cols].values.astype(np.float32)
        label_data = sequence_df[label_cols].values.astype(np.float32)

        return {
            'depth_img': depth_tensor,
            'rgb_img': rgb_tensor,
            'state': torch.from_numpy(state_data),
            'label': torch.from_numpy(label_data)
        }

if __name__ == '__main__':
    # --- Test Script ---
    print("--- Testing HumanoidNavDataset ---")

    # Define directories and parameters
    PROCESSED_DATA_DIR = 'data/processed'
    SEQ_LENGTH = 16
    BATCH_SIZE = 4

    # Check if processed data exists
    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        print(f"Error: Processed data directory '{PROCESSED_DATA_DIR}' is empty or does not exist.")
        print("Please run 'generate_dummy_data.py' and 'preprocess_data.py' first.")
    else:
        all_csv_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')]

        # Define standard transformations
        img_transforms = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            # Add normalization if needed, e.g., transforms.Normalize(mean=[...], std=[...])
        ])

        # 1. Test with Depth images only
        print("\n1. Testing with Depth images only...")
        try:
            dataset_depth = HumanoidNavDataset(
                processed_data_dir=PROCESSED_DATA_DIR,
                csv_files=all_csv_files,
                sequence_length=SEQ_LENGTH,
                use_rgb=False,
                transform=img_transforms
            )
            sample = dataset_depth[0]
            print(f"  Sample 0 shapes:")
            print(f"    depth_img: {sample['depth_img'].shape}")
            print(f"    rgb_img:   {sample['rgb_img'].shape} (empty as expected)")
            print(f"    state:     {sample['state'].shape}")
            print(f"    label:     {sample['label'].shape}")
            assert sample['depth_img'].shape == (SEQ_LENGTH, 1, 480, 640)
            assert sample['state'].shape == (SEQ_LENGTH, 2)
            assert sample['label'].shape == (SEQ_LENGTH, 3)
            print("  Shape tests passed!")
        except Exception as e:
            print(f"  Test failed: {e}")

        # 2. Test with Depth + RGB images
        print("\n2. Testing with Depth and RGB images...")
        try:
            dataset_full = HumanoidNavDataset(
                processed_data_dir=PROCESSED_DATA_DIR,
                csv_files=all_csv_files,
                sequence_length=SEQ_LENGTH,
                use_rgb=True,
                transform=img_transforms
            )
            sample = dataset_full[0]
            print(f"  Sample 0 shapes:")
            print(f"    depth_img: {sample['depth_img'].shape}")
            print(f"    rgb_img:   {sample['rgb_img'].shape}")
            print(f"    state:     {sample['state'].shape}")
            print(f"    label:     {sample['label'].shape}")
            assert sample['depth_img'].shape == (SEQ_LENGTH, 1, 480, 640)
            assert sample['rgb_img'].shape == (SEQ_LENGTH, 3, 480, 640)
            print("  Shape tests passed!")
        except Exception as e:
            print(f"  Test failed: {e}")

        # 3. Test with DataLoader
        print("\n3. Testing with DataLoader...")
        try:
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset_full, batch_size=BATCH_SIZE, shuffle=True)
            batch_sample = next(iter(dataloader))
            print(f"  Batch shapes:")
            print(f"    depth_img: {batch_sample['depth_img'].shape}")
            print(f"    rgb_img:   {batch_sample['rgb_img'].shape}")
            print(f"    state:     {batch_sample['state'].shape}")
            print(f"    label:     {batch_sample['label'].shape}")
            assert batch_sample['depth_img'].shape == (BATCH_SIZE, SEQ_LENGTH, 1, 480, 640)
            assert batch_sample['rgb_img'].shape == (BATCH_SIZE, SEQ_LENGTH, 3, 480, 640)
            print("  DataLoader test passed!")
        except Exception as e:
            print(f"  Test failed: {e}")

        print("\n--- HumanoidNavDataset tests completed! ---")
