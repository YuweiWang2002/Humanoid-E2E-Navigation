"""
This script implements an online/incremental normalization manager.
"""

import torch
import os
import json
from torchvision import transforms

class NormalizationManager:
    """Manages the calculation and persistence of normalization statistics (mean, std)
    for a dataset in an online, incremental fashion using Welford's algorithm.
    """
    def __init__(self, stats_dim, file_path='normalization_stats.json'):
        """
        Initializes the manager.

        Args:
            stats_dim (int): The dimension of the statistics to be tracked (e.g., 1 for grayscale, 3 for RGB, 5 for state vector).
            file_path (str): The path to the JSON file for loading/saving stats.
        """
        self.file_path = file_path
        self.stats_dim = stats_dim
        self.n = 0
        self.mean = torch.zeros(stats_dim)
        self.M2 = torch.zeros(stats_dim)
        self.std = torch.ones(stats_dim)  # Start with std=1 to avoid division by zero
        self.load()

    def update(self, new_data_batch):
        """
        Updates the running statistics with a new batch of data using Welford's online algorithm.

        Args:
            new_data_batch (torch.Tensor): A tensor of new data, shape (N, stats_dim).
        """
        batch_n = new_data_batch.shape[0]
        if batch_n == 0:
            return

        for i in range(batch_n):
            self.n += 1
            x = new_data_batch[i]
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean  # New delta after mean update
            self.M2 += delta * delta2
        
        if self.n > 1:
            self.std = torch.sqrt(self.M2 / (self.n - 1)) # Use n-1 for sample standard deviation
            # Prevent std from being zero
            self.std[self.std < 1e-6] = 1e-6

    def get_transform(self):
        """Returns a torchvision Normalize transform object with the current stats."""
        return transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())

    def save(self):
        """Saves the current statistics to the specified JSON file."""
        stats = {
            'n': self.n,
            'mean': self.mean.tolist(),
            'M2': self.M2.tolist()
        }
        with open(self.file_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Normalization stats saved to {self.file_path}")

    def load(self):
        """Loads statistics from the specified JSON file if it exists."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    stats = json.load(f)
                self.n = stats['n']
                self.mean = torch.tensor(stats['mean'], dtype=torch.float32)
                self.M2 = torch.tensor(stats['M2'], dtype=torch.float32)
                if self.n > 1:
                    self.std = torch.sqrt(self.M2 / (self.n - 1))
                    self.std[self.std < 1e-6] = 1e-6
                print(f"Normalization stats loaded from {self.file_path}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load or parse {self.file_path}. Starting with fresh stats. Error: {e}")
                self._reset()
        else:
            print(f"No normalization stats file found at {self.file_path}. Starting with fresh stats.")

    def _reset(self):
        """Resets the statistics to their initial state."""
        self.n = 0
        self.mean = torch.zeros(self.stats_dim)
        self.M2 = torch.zeros(self.stats_dim)
        self.std = torch.ones(self.stats_dim)
