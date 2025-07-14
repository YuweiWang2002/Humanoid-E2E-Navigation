"""MLP head for processing state information.

This file defines an MLP head for processing non-image state data,
such as localization, orientation, and target coordinates.
"""
import torch
from torch import nn


class MLPHead(nn.Module):
    """This class defines a simple MLP head for state feature extraction."""

    def __init__(self, input_dim=2, hidden_dims=[32, 64], output_dim=64):
        """
        Initializes the MLPHead.

        Args:
            input_dim (int): The dimension of the input state vector.
                             Defaults to 2 for (distance_to_target, angle_to_target).
            hidden_dims (list): A list of integers specifying the size of each hidden layer.
            output_dim (int): The dimension of the output feature vector. Defaults to 64.
        """
        super(MLPHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.ReLU(inplace=True))  # Add a final activation

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the MLP head.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim) or
                              (batch_size, time_sequence, input_dim).

        Returns:
            torch.Tensor: Output feature tensor with shape (batch_size, output_dim) or
                          (batch_size, time_sequence, output_dim).
        """
        # The MLP can handle both (B, F) and (B, T, F) inputs automatically.
        # If input is (B, T, F), the linear layers will be applied to the last dimension F.
        return self.mlp(x)

    def count_params(self):
        """Return back how many params MLP_head has."""
        return sum(param.numel() for param in self.parameters())


if __name__ == "__main__":
    # --- Test MLPHead ---
    print("--- Testing MLPHead ---")

    # Test parameters
    batch_size_test = 4
    time_seq_test = 16
    input_dim_test = 2   # As per your request
    output_dim_test = 64  # The chosen output feature dimension

    mlp_model = MLPHead(input_dim=input_dim_test, output_dim=output_dim_test)
    print(f"MLP Model Structure:\n{mlp_model}")
    print(f"MLP Model params: {mlp_model.count_params()}\n")

    # Test with a non-sequential input (e.g., for a pure MLP model)
    dummy_input_non_seq = torch.randn(batch_size_test, input_dim_test)
    output_non_seq = mlp_model(dummy_input_non_seq)
    print(f"Input shape (non-sequential): {dummy_input_non_seq.shape}")
    print(f"Output shape (non-sequential): {output_non_seq.shape}")
    assert output_non_seq.shape == (batch_size_test, output_dim_test)

    # Test with a sequential input (e.g., for an RNN-based model)
    dummy_input_seq = torch.randn(batch_size_test, time_seq_test, input_dim_test)
    output_seq = mlp_model(dummy_input_seq)
    print(f"\nInput shape (sequential): {dummy_input_seq.shape}")
    print(f"Output shape (sequential): {output_seq.shape}")
    assert output_seq.shape == (batch_size_test, time_seq_test, output_dim_test)

    print("\n--- MLPHead tests passed! ---")