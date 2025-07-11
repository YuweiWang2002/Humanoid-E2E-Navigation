"""This script defines all the models except NCP used in this work."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
# Add the parent directory to the Python path
# This assumes cnn_head.py is in the 'nets/' directory, and 'utils.py' is in its parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import _image_standardization
from nets.cnn_head import ConvolutionHead_Nvidia, ConvolutionHead_ResNet, ConvolutionHead_AlexNet



class Convolution_Model(nn.Module):
    """This class defines CNN baseline."""

    def __init__(self, img_dim, a_dim, use_cuda=True):
        """Initialize the object."""
        super(Convolution_Model, self).__init__()
        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")
        self.loss = nn.MSELoss(reduction='mean')

        # Save image dimension information
        self.img_channel = img_dim[0]
        self.img_height = img_dim[1]
        self.img_width = img_dim[2]

        self.conv = nn.Sequential(
            # Output: (24, ~238, ~318)
            nn.Conv2d(self.img_channel, 24, kernel_size=5, stride=2, bias=True),
            nn.ReLU(inplace=True),
            # Output: (36, ~118, ~158)
            nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=True),
            nn.ReLU(inplace=True),
            # Output: (48, ~58, ~78)
            nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=True),
            nn.ReLU(inplace=True),
            # New: Aggressive downsampling here
            # Output: (64, ~28, ~38)
            nn.Conv2d(48, 64, kernel_size=3, stride=2, bias=True), # Increased stride
            nn.ReLU(inplace=True),
            # New: More downsampling
            # Output: (64, ~13, ~18) - Check exact calculation due to small dims
            nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=True), # Increased stride
            nn.ReLU(inplace=True)
        )
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        # Dynamically calculate the input feature number for the linear1 layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.img_channel, self.img_height, self.img_width)
            output_features = self.conv(dummy_input)
            self.flattened_features_size = output_features.view(output_features.size(0), -1).size(1)
            print(f"Convolution_Model - Calculated flattened_features_size for linear1: {self.flattened_features_size}")

        self.linear1 = nn.Linear(self.flattened_features_size, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 10)
        self.linear4 = nn.Linear(10, a_dim)
        self.a_dim = a_dim

        self.exp_factor = 0.1
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = _image_standardization(x)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)
        x = self.conv(x)
        x = x.view(-1, self.flattened_features_size)
        x = self.dropout1(x)
        x = F.relu(self.linear1(x))
        x = self.dropout2(x)
        x = F.relu(self.linear2(x))
        x = self.dropout3(x)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def criterion(self, a_imitator, a_exp):
        loss = self.loss(a_imitator, a_exp)
        return loss

    def weighted_criterion(self, a_imitator, a_exp):
        assert self.exp_factor >= 0
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor)
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def release(self, sdir):
        torch.save(self.state_dict(), sdir + "policy_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + "policy_optim.pth")

    def load(self, ldir):
        try:
            self.load_state_dict(
                torch.load(
                    ldir + "policy_model.pth",
                    map_location=torch.device('cpu')
                )
            )
            self.optimizer.load_state_dict(
                torch.load(
                    ldir + "policy_optim.pth",
                    map_location=torch.device('cpu')
                )
            )
            print("load parameters are in" + ldir)
            return True
        except Exception as e:
            print(f"parameters are not loaded. Error: {e}")
            return False

    def count_params(self):
        num_params = sum(param.numel() for param in self.parameters())
        return num_params

    def nn_structure(self):
        dict_layer = {i: self._modules[i] for i in self._modules.keys()}
        return dict_layer


class GRU_Model(nn.Module):
    """This class defines GRU model, layer is equal to 1."""

    def __init__(self, conv_head, time_step=16, hidden_size=64,
                 output=3, use_cuda=True):
        """
        Initializes the GRU_Model.

        Args:
            conv_head (nn.Module): The CNN head module for feature extraction.
                                   Its output features will determine input_size.
            time_step (int, optional): The sequence length for the GRU. Defaults to 16.
            hidden_size (int, optional): The number of features in the GRU's hidden state. Defaults to 64.
            output (int, optional): The dimension of the model's output (e.g., 3 for velX, velY, vel_yaw). Defaults to 3.
            use_cuda (bool, optional): Whether to use CUDA if available. Defaults to True.
        """
        super(GRU_Model, self).__init__()
        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")

        self.loss = nn.MSELoss(reduction='mean')
        self.exp_factor = 0.1  # Weight factor for the weighted loss
        self.num_params = 0    # To store the total number of trainable parameters

        self.conv_head = conv_head  # CNN layer (feature extractor) before the GRU
        self.time_step = time_step

        # Dynamically determine the input_size from the conv_head's total_features.
        # This makes the GRU_Model more robust to changes in the conv_head's output dimensions.
        if hasattr(self.conv_head, 'total_features'):
            self.input_size = self.conv_head.total_features
        else:
            # Fallback if conv_head doesn't explicitly define total_features.
            # You might need to run a dummy forward pass on conv_head to get its output shape
            # if total_features is not a direct attribute.
            # For current cnn_head implementation, total_features is present.
            # A more robust check might involve:
            # with torch.no_grad():
            #     dummy_img = torch.zeros(1, conv_head.img_channel, conv_head.img_height, conv_head.img_width).to(self.device)
            #     dummy_output = conv_head(dummy_img).squeeze(0) # remove time_sequence dim for this check
            #     self.input_size = dummy_output.shape[-1]
            print("Warning: 'total_features' not found in conv_head. Using provided input_size.")
            # If `total_features` is not guaranteed, you might need to pass `input_size` explicitly or calculate it.
            # For now, let's assume `conv_head` has `total_features` or `input_size` will be provided correctly.
            self.input_size = 128 # Default value if not set via conv_head attribute

        self.hidden_size = hidden_size  # The number of features in the GRU's hidden state
        self.output = output            # The dimension of the final output (e.g., steering, throttle, brake)

        # GRU layer: batch_first=True means input/output tensors are (batch_size, sequence_length, features)
        self.GRU = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        
        # Linear layer to map the GRU's hidden_size output to the desired action dimension
        self.linear = nn.Linear(self.hidden_size, self.output)

        # The optimizer should be defined after all layers are initialized
        # so that self.parameters() can correctly collect all learnable parameters.
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        """
        Defines the forward pass of the GRU model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, time_sequence, channel, height, width).

        Returns:
            torch.Tensor: Output tensor with predicted actions, shape (batch_size, time_sequence, output).
        """
        # Store original batch size for reshaping output
        batch_size = x.shape[0]

        # Process input through the CNN head to extract features
        # Output shape: (batch_size, time_sequence, extracted_features)
        x = self.conv_head(x)

        # Pass the extracted features through the GRU layer
        # h_0 (initial hidden state) defaults to zeros if not provided
        # x_out shape: (batch_size, time_sequence, hidden_size)
        # _ represents the final hidden state (h_T), which is not used in this forward pass
        x_out, _ = self.GRU(x)

        # Reshape the GRU output for the linear layer
        # .contiguous() is used to ensure memory contiguity before .view()
        x_out = x_out.contiguous().view(-1, self.hidden_size)

        # Map the hidden state output to the final action dimension
        x_out = self.linear(x_out)

        # Reshape the output back to (batch_size, time_sequence, output)
        x_out = x_out.view(batch_size, self.time_step, self.output)
        return x_out

    def evaluate_on_single_sequence(self, x, hidden_state=None):
        """
        Evaluates the model on a single sequence (e.g., for test or validation).
        This method allows for passing and returning hidden states for sequential inference.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, time_sequence, channel, height, width).
                              Typically, batch_size is 1 for single sequence evaluation.
            hidden_state (torch.Tensor, optional): Initial hidden state for the GRU.
                                                   Shape: (num_layers * num_directions, batch_size, hidden_size).
                                                   Defaults to None (initialized to zeros).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Predicted actions, shape (1, time_sequence, output).
                - torch.Tensor: The final hidden state of the GRU.
        """
        # Process input through the CNN head
        # Output shape: (batch_size, time_sequence, extracted_features)
        x = self.conv_head(x)

        # Initialize hidden state to zeros if not provided
        if hidden_state is None:
            # GRU's hidden state shape: (num_layers * num_directions, batch_size, hidden_size)
            # For a single-layer GRU (default), num_layers * num_directions is 1.
            hidden_state = torch.zeros((1, x.shape[0], self.hidden_size),
                                       device=x.device)

        # Pass through GRU, returning output and updated hidden state
        # result shape: (batch_size, time_sequence, hidden_size)
        # hidden_state_out shape: (1, batch_size, hidden_size)
        result, hidden_state_out = self.GRU(x, hidden_state)

        # Reshape for the linear layer
        result = result.contiguous().view(-1, self.hidden_size)

        # Map to action dimension
        result = self.linear(result)

        # Add an unsqueeze dimension for consistency (e.g., if batch_size was 1, make it explicit)
        # If the input batch_size is always 1 during evaluation, this ensures (1, time_step, output)
        result = result.view(x.shape[0], self.time_step, self.output) # Reshape back to (Batch, Time, Output)
        
        # Consider if you want the output to be (1, 16, 3) for `evaluate_on_single_sequence` 
        # or just (16, 3) if batch_size is always 1.
        # The line `result = torch.unsqueeze(result, dim=0)` in your original code
        # would be for a scenario where `result` was `(time_step, output)` and you needed a batch dimension.
        # Given `result = result.view(x.shape[0], self.time_step, self.output)` already includes batch_size,
        # `torch.unsqueeze(result, dim=0)` would add an extra redundant dimension if `x.shape[0]` is already 1.
        # So, the current `result.view` is more general.

        return result, hidden_state_out

    def criterion(self, a_imitator, a_exp):
        """
        Calculates the standard Mean Squared Error (MSE) loss.

        Args:
            a_imitator (torch.Tensor): Predicted actions from the model.
            a_exp (torch.Tensor): Ground truth (expert) actions.

        Returns:
            torch.Tensor: The MSE loss value.
        """
        loss = self.loss(a_imitator, a_exp)
        return loss

    def weighted_criterion(self, a_imitator, a_exp):
        """
        Calculates a weighted MSE loss, where weights are based on the absolute value of expert actions.
        This can be useful to emphasize learning in critical (e.g., larger steering angle) situations.

        Args:
            a_imitator (torch.Tensor): Predicted actions from the model.
            a_exp (torch.Tensor): Ground truth (expert) actions.

        Returns:
            torch.Tensor: The weighted MSE loss value.
        """
        assert self.exp_factor >= 0, "exp_factor must be non-negative for weighted loss."
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor) # Weights increase exponentially with action magnitude
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def release(self, sdir):
        """
        Saves the trained model's state dictionary and optimizer's state dictionary.

        Args:
            sdir (str): Directory path to save the model files.
        """
        torch.save(self.state_dict(), sdir + "policy_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + "policy_optim.pth")

    def load(self, ldir):
        """
        Loads a trained model's state dictionary and optimizer's state dictionary.

        Args:
            ldir (str): Directory path to load the model files from.

        Returns:
            bool: True if parameters were loaded successfully, False otherwise.
        """
        try:
            self.load_state_dict(
                torch.load(ldir + "policy_model.pth",
                           map_location=torch.device('cpu'))
            )
            self.optimizer.load_state_dict(
                torch.load(ldir + "policy_optim.pth",
                           map_location=torch.device('cpu'))
            )
            print("Loaded parameters from: " + ldir)
            return True
        except Exception as e: # Catch specific exception for better debugging
            print(f"Parameters could not be loaded. Error: {e}")
            return False

    def count_params(self):
        """
        Counts the total number of learnable parameters in the network.

        Returns:
            int: Total number of learnable parameters.
        """
        self.num_params = sum(param.numel() for param in self.parameters())
        return self.num_params

    def nn_structure(self):
        """
        Returns a dictionary containing the model's direct child modules.

        Returns:
            dict: A dictionary mapping module names to their instances.
        """
        dict_layer = {i: self._modules[i] for i in self._modules.keys()}
        return dict_layer


class LSTM_Model(nn.Module):
    """This class defines the LSTM model, with num_layers set to 1."""

    def __init__(self, conv_head, time_step=16, hidden_size=64,
                 output=3, use_cuda=True):
        """
        Initializes the LSTM_Model.

        Args:
            conv_head (nn.Module): The CNN head module for feature extraction.
                                   Its output features will determine input_size.
            time_step (int, optional): The sequence length for the LSTM. Defaults to 16.
            hidden_size (int, optional): The number of features in the LSTM's hidden state (h and c). Defaults to 64.
            output (int, optional): The dimension of the model's output (e.g., 3 for velX, velY, vel_yaw). Defaults to 3.
            use_cuda (bool, optional): Whether to use CUDA if available. Defaults to True.
        """
        super(LSTM_Model, self).__init__()

        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")
        self.loss = nn.MSELoss(reduction='mean')
        self.exp_factor = 0.1  # Weight factor for the weighted loss
        self.num_params = 0    # To store the total number of trainable parameters

        self.conv_head = conv_head   # CNN layer (feature extractor) before the LSTM
        self.time_step = time_step

        # Dynamically determine the input_size from the conv_head's total_features.
        # This makes the LSTM_Model more robust to changes in the conv_head's output dimensions.
        if hasattr(self.conv_head, 'total_features'):
            self.input_size = self.conv_head.total_features
        else:
            # Fallback if conv_head doesn't explicitly define total_features.
            print("Warning: 'total_features' not found in conv_head. Using provided input_size.")
            self.input_size = 128 # Default value if not set via conv_head attribute

        self.hidden_size = hidden_size  # The number of features in the LSTM's hidden state (h and c)
        self.output = output            # The dimension of the final output (e.g., steering, throttle, brake)

        # LSTM layer: batch_first=True means input/output tensors are (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            batch_first=True)  # default: num_layers = 1
        
        # Linear layer to map the LSTM's hidden_size output to the desired action dimension
        # Based on your comment 'wyw: linear(64,3) -> linear(64,3) 修改输出维度为3' and 'wyw: output = 3'
        # ensure self.output is correctly set and used here.
        self.linear = nn.Linear(self.hidden_size, self.output)

        # The optimizer should be defined after all layers are initialized
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):  # No need to define h_0 and c_0 explicitly, LSTM handles default zeros.
        """
        Defines the forward pass for the LSTM model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, time_sequence, channel, height, width).

        Returns:
            torch.Tensor: Output tensor with predicted actions, shape (batch_size, time_sequence, output).
        """
        batch_size = x.shape[0]
        # Process input through the CNN head to extract features
        # Output shape: (batch_size, time_sequence, extracted_features)
        x = self.conv_head(x)

        # Pass the extracted features through the LSTM layer
        # (h_0, c_0 default to zeros if not provided),
        # (h_T, c_T) represent the cell state/hidden state of the last time step.
        # x_out shape: (batch_size, time_step, hidden_size)
        x_out, _ = self.lstm(x)

        # Reshape the LSTM output for the linear layer
        # .contiguous() ensures memory contiguity before .view()
        x_out = x_out.contiguous().view(-1, self.hidden_size)
        
        # Map the hidden state output to the final action dimension
        x_out = self.linear(x_out)

        # Reshape the output back to (batch_size, time_sequence, output)
        x_out = x_out.view(batch_size, self.time_step, self.output)
        return x_out

    def evaluate_on_single_sequence(self, x, hidden_state=None):
        """
        Evaluates the model on a single sequence sequentially (e.g., for valid or test).
        This method allows for passing and returning hidden states (h and c) for sequential inference.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, time_sequence, channel, height, width).
                              Typically, batch_size is 1 for single sequence evaluation.
            hidden_state (tuple, optional): Initial hidden state (h0, c0) for the LSTM.
                                            Each tensor in the tuple has shape:
                                            (num_layers * num_directions, batch_size, hidden_size).
                                            Defaults to None (initialized to zeros).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Predicted actions, shape (batch_size, time_sequence, output).
                - tuple: The final hidden state (h_n, c_n) of the LSTM.
        """
        # Process input through the CNN head
        # Output shape: (batch_size, time_sequence, extracted_features)
        x = self.conv_head(x)

        if hidden_state is None:
            # Initialize hidden state (h0, c0) to zeros if not provided.
            # Each tensor has shape: (num_layers * num_directions, batch_size, hidden_size)
            # For a single-layer LSTM, num_layers * num_directions is 1.
            hidden_state = (torch.zeros((1, x.shape[0], self.hidden_size), device=x.device),
                            torch.zeros((1, x.shape[0], self.hidden_size), device=x.device))

        # Pass through LSTM, returning output and updated hidden state (h_n, c_n)
        # result shape: (batch_size, time_sequence, hidden_size)
        # hidden_state_out is a tuple (h_n, c_n)
        result, hidden_state_out = self.lstm(x, hidden_state)

        # Reshape for the linear layer
        # In this case, N=batch_size, L=time_step, H_out=hidden_size -> (N*L, H_out)
        result = result.contiguous().view(-1, self.hidden_size)
        
        # Map to action dimension
        result = self.linear(result)

        # Reshape the output back to (batch_size, time_sequence, output)
        result = result.view(x.shape[0], self.time_step, self.output)
        
        return result, hidden_state_out

    def criterion(self, a_imitator, a_exp):
        """
        Calculates the standard Mean Squared Error (MSE) loss.

        Args:
            a_imitator (torch.Tensor): Predicted actions from the model.
            a_exp (torch.Tensor): Ground truth (expert) actions.

        Returns:
            torch.Tensor: The MSE loss value.
        """
        loss = self.loss(a_imitator, a_exp)
        return loss

    def weighted_criterion(self, a_imitator, a_exp):
        """
        Calculates a weighted MSE loss, where weights are based on the absolute value of expert actions.
        This can be useful to emphasize learning in critical (e.g., larger steering angle) situations.

        Args:
            a_imitator (torch.Tensor): Predicted actions from the model.
            a_exp (torch.Tensor): Ground truth (expert) actions.

        Returns:
            torch.Tensor: The weighted MSE loss value.
        """
        assert self.exp_factor >= 0, "exp_factor must be non-negative for weighted loss."
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor) # Weights increase exponentially with action magnitude
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def release(self, sdir):
        """
        Saves the trained model's state dictionary and optimizer's state dictionary.

        Args:
            sdir (str): Directory path to save the model files.
        """
        torch.save(self.state_dict(), sdir + "policy_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + "policy_optim.pth")

    def load(self, ldir):
        """
        Loads a trained model's state dictionary and optimizer's state dictionary.

        Args:
            ldir (str): Directory path to load the model files from.

        Returns:
            bool: True if parameters were loaded successfully, False otherwise.
        """
        try:
            self.load_state_dict(
                torch.load(
                    ldir + "policy_model.pth",
                    map_location=torch.device('cpu')
                )
            )
            self.optimizer.load_state_dict(
                torch.load(
                    ldir + "policy_optim.pth",
                    map_location=torch.device('cpu')
                )
            )
            print("Loaded parameters from: " + ldir)
            return True
        except Exception as e: # Catch specific exception for better debugging
            print(f"Parameters could not be loaded. Error: {e}")
            return False

    def count_params(self):
        """
        Counts the total number of learnable parameters in the network.

        Returns:
            int: Total number of learnable parameters.
        """
        self.num_params = sum(param.numel() for param in self.parameters())
        return self.num_params

    def nn_structure(self):
        """
        Returns a dictionary containing the model's direct child modules.

        Returns:
            dict: A dictionary mapping module names to their instances.
        """
        dict_layer = {i: self._modules[i] for i in self._modules.keys()}
        return dict_layer


class CTGRU_Model(nn.Module):
    """
    This class defines a Continuous-Time Gated Recurrent Unit (CT-GRU) model.
    It incorporates multiple "traces" (M) with varying time constants to capture
    information at different temporal scales.
    """

    def __init__(self, num_units, conv_head,
                 M=8, time_step=16, output=3, use_cuda=True):
        """
        Initializes the CTGRU_Model.

        Args:
            num_units (int): The number of hidden units (neurons) in the CT-GRU. This is the 'hidden_size' of the main hidden state.
            conv_head (nn.Module): The CNN head module for feature extraction.
                                   Its output features will determine the input size for the CT-GRU.
            M (int, optional): The number of traces (time scales) in the CT-GRU. Defaults to 8.
            time_step (int, optional): The sequence length for the CT-GRU. Defaults to 16.
            output (int, optional): The dimension of the model's final output (e.g., 3 for velX, velY, vel_yaw). Defaults to 3.
            use_cuda (bool, optional): Whether to use CUDA if available. Defaults to True.
        """
        super(CTGRU_Model, self).__init__()
        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")
        self.loss = nn.MSELoss(reduction='mean')
        self.exp_factor = 0.1 # Initialize exp_factor for weighted loss

        self.conv_head = conv_head
        self._num_units = num_units  # The hidden size for each time step
        self.M = M                   # Number of memory traces (time scales)

        # Initialize logarithmic time constants for the traces
        ln_tau_table = np.empty(self.M)
        tau = 1.0 # Initial time constant
        for i in range(self.M):
            ln_tau_table[i] = np.log(tau)
            tau = tau * (10 ** 0.5) # tau_i+1 = 10^0.5 * tau_i, providing high fidelity time scales

        self.ln_tau_table = torch.tensor(
            ln_tau_table, dtype=torch.float32, device=self.device
        )

        self.time_step = time_step
        self.output = output

        # Image interval, 0.04 for training, 0.2 for simulation test
        # Made this a member variable for flexibility
        self.delta_t = torch.tensor(0.04, device=self.device)

        # Dynamically determine the input feature number from the conv_head
        if hasattr(self.conv_head, 'total_features'):
            self.feature_number = self.conv_head.total_features
        else:
            # Fallback or a more robust check if total_features is not a direct attribute
            print("Warning: 'total_features' not found in conv_head. Please ensure conv_head provides correct feature number.")
            # You might need to infer it from conv_head's output with a dummy pass
            # For now, let's assume `conv_head` has `total_features` or `input_size` will be provided correctly.
            # For now, let's assume `conv_head` has `total_features` or `input_size` will be provided correctly.
            # If ConvolutionHead_Nvidia is used, its `total_features` attribute will be available.
            self.feature_number = 128 # Default based on common scenarios or previous input_size

        # Linear layers for CT-GRU gates
        # linear_r for retrieval scale (r_ki)
        self.linear_r = nn.Linear(self.feature_number + self._num_units, self._num_units * M)
        # linear_q for relevant event signals (q_k)
        self.linear_q = nn.Linear(self.feature_number + self._num_units, self._num_units)
        # linear_s for storage scale (s_ki)
        self.linear_s = nn.Linear(self.feature_number + self._num_units, self._num_units * M)

        # Final linear layer to map the hidden state to the output action space
        self.linear = nn.Linear(self._num_units, self.output)

        self.softmax_r = nn.Softmax(dim=2) # Softmax along the M (trace) dimension
        self.softmax_s = nn.Softmax(dim=2) # Softmax along the M (trace) dimension
        self.tanh = nn.Tanh() # Activation for q_k

        self.num_params = 0 # To store the total number of trainable parameters
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=1e-4,
            weight_decay=0 # weight_decay defaults to 0
        )

    @property
    def state_size(self):
        """
        Defines the total size of the internal state (h_hat), which is (num_units * M).
        """
        return self._num_units * self.M

    @property
    def output_size(self):
        """
        Defines the size of the main hidden output (summed across traces), which is num_units.
        """
        return self._num_units

    def update(self, x, h_hat, delta_t):
        """
        Updates the states of the CT-GRU for one time interval.

        Args:
            x (torch.Tensor): Input at the current time step, shape (batch_size, feature_number).
            h_hat (torch.Tensor): The current internal memory traces, shape (batch_size, num_units, M).
            delta_t (torch.Tensor): Time interval between current and next input, shape (scalar).

        Returns:
            tuple: A tuple containing:
                - h_hat_next (torch.Tensor): Updated internal memory traces, shape (batch_size, num_units, M).
                - hidden_state (torch.Tensor): The current main hidden state (summed across traces), shape (batch_size, num_units).
        """
        # Sum across all traces to get the current aggregated hidden state h
        # h shape: (batch_size, num_units)
        h = h_hat.sum(dim=2)

        # 1. Determine retrieval scale and weighting (r_ki)
        # Concatenate current input (x) and aggregated hidden state (h)
        # fused_input shape: (batch_size, feature_number + num_units)
        fused_input = torch.cat([x, h], dim=1)

        # Calculate ln_tau_r
        # ln_tau_r_raw shape: (batch_size, num_units * M)
        ln_tau_r_raw = self.linear_r(fused_input)
        # Reshape to (batch_size, num_units, M) to align with h_hat and ln_tau_table
        ln_tau_r = ln_tau_r_raw.view(-1, self._num_units, self.M)

        # Calculate softmax input for r_ki (negative squared difference from target time constants)
        sf_input_r = -torch.square(ln_tau_r - self.ln_tau_table)
        # Apply softmax to get normalized retrieval weights r_ki
        # r_ki shape: (batch_size, num_units, M)
        r_ki = self.softmax_r(sf_input_r)

        # 2. Determine relevant event signals (q_k)
        # Calculate context-aware input for q_k: sum of (r_ki * h_hat) across traces
        # This selectively retrieves information from h_hat based on r_ki
        fused_q_input_context = torch.sum(r_ki * h_hat, dim=2) # Shape: (batch_size, num_units)
        fused_q_input = torch.cat([x, fused_q_input_context], dim=1) # Shape: (batch_size, feature_number + num_units)

        # Calculate q_k (candidate for new memory trace)
        # q_k_raw shape: (batch_size, num_units)
        q_k_raw = self.linear_q(fused_q_input)
        # Apply Tanh activation and reshape for broadcasting (batch_size, num_units, 1)
        q_k = self.tanh(q_k_raw).reshape(-1, self._num_units, 1)

        # 3. Determine storage scale and weighting (s_ki)
        # Calculate ln_tau_s (similar to ln_tau_r)
        # ln_tau_s_raw shape: (batch_size, num_units * M)
        ln_tau_s_raw = self.linear_s(fused_input)
        # Reshape to (batch_size, num_units, M)
        ln_tau_s = ln_tau_s_raw.view(-1, self._num_units, self.M)

        # Calculate softmax input for s_ki
        sf_input_s = -torch.square(ln_tau_s - self.ln_tau_table)
        # Apply softmax to get normalized storage weights s_ki
        # s_ki shape: (batch_size, num_units, M)
        s_ki = self.softmax_s(sf_input_s)

        # 4. Calculate h_hat_next (update internal memory traces)
        # The core CT-GRU update rule:
        # (1 - s_ki) * h_hat : retains old information
        # s_ki * q_k          : stores new information
        # torch.exp(-delta_t / (self.ln_tau_table + 1e-7)) : decays information based on time constants
        h_hat_next = ((1 - s_ki) * h_hat + s_ki * q_k) * \
                     torch.exp(-delta_t / (self.ln_tau_table.exp() + 1e-7)) # Use exp() to convert ln_tau_table back to tau
                                                                              # Added 1e-7 for numerical stability to avoid division by zero if tau somehow becomes zero

        # Combine time-scales to get the main hidden state for the current time step
        # hidden_state shape: (batch_size, num_units)
        hidden_state = torch.sum(h_hat_next, dim=2)

        return h_hat_next, hidden_state

    def forward(self, x):
        """
        Defines the forward pass of the CT-GRU model.
        Processes a sequence of inputs and produces a sequence of outputs.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, time_sequence, channel, height, width).

        Returns:
            torch.Tensor: Output tensor with predicted actions, shape (batch_size, time_sequence, output).
        """
        # Process input through the CNN head
        # Output shape: (batch_size, time_sequence, feature_number)
        x = self.conv_head(x)

        # Initialize internal memory traces (h_hat) to zeros
        # h_hat shape: (batch_size, num_units, M)
        h_hat = torch.zeros((x.shape[0], self._num_units, self.M), device=x.device)

        outputs = []
        for t in range(self.time_step):
            # Get input for the current time step
            inputs = x[:, t, :] # Shape: (batch_size, feature_number)
            
            # Update the CT-GRU states
            h_hat, hidden_state = self.update(inputs, h_hat, self.delta_t)
            
            # Append the main hidden state to outputs
            outputs.append(hidden_state)

        # Stack outputs along the time_step dimension
        # outputs shape: (batch_size, time_sequence, num_units)
        outputs = torch.stack(outputs, dim=1)

        # Reshape for the final linear layer
        outputs = outputs.contiguous().view(-1, self._num_units)
        
        # Map hidden states to action outputs
        outputs = self.linear(outputs)

        # Reshape to final output format (batch_size, time_sequence, output)
        outputs = outputs.view(-1, self.time_step, self.output)
        return outputs

    def evaluate_on_single_sequence(self, x, hidden_state=None):
        """
        Evaluates the model on a single sequence sequentially (e.g., for validation or test).
        This method is designed for processing a sequence step-by-step and
        maintaining the internal memory state (h_hat) across steps.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, time_sequence, channel, height, width).
                              Typically, batch_size is 1 for single sequence evaluation.
            hidden_state (torch.Tensor, optional): Initial internal memory traces (h_hat).
                                                   Shape: (batch_size, num_units, M).
                                                   Defaults to None (initialized to zeros).

        Returns:
            tuple: A tuple containing:
                - results (torch.Tensor): Predicted actions, shape (batch_size, time_sequence, output).
                - hidden_state (torch.Tensor): The final internal memory traces (h_hat) after processing the sequence.
        """
        # Process input through the CNN head
        # Output shape: (batch_size, time_sequence, feature_number)
        x = self.conv_head(x)
        
        results = []
        if hidden_state is None:
            # Initialize internal memory traces (h_hat) to zeros if not provided
            hidden_state = torch.zeros(
                (x.shape[0], self._num_units, self.M), device=x.device
            )

        for t in range(self.time_step):
            # Get input for the current time step
            inputs = x[:, t, :] # Shape: (batch_size, feature_number)
            
            # Update the CT-GRU states
            # Note: `update` returns (h_hat_next, hidden_state_current_step)
            hidden_state, current_step_hidden_output = self.update(inputs, hidden_state, self.delta_t)
            
            # Append the main hidden output for this step to results
            results.append(current_step_hidden_output)

        # Stack results along the time_step dimension
        # results_stacked shape: (batch_size, time_sequence, num_units)
        results_stacked = torch.stack(results, dim=1)
        
        # Reshape for the final linear layer
        results_linear_input = results_stacked.contiguous().view(-1, self._num_units)
        
        # Map hidden states to action outputs
        results_final = self.linear(results_linear_input)

        # Reshape to final output format (batch_size, time_sequence, output)
        results_final = results_final.view(x.shape[0], self.time_step, self.output)
        
        return results_final, hidden_state

    def criterion(self, a_imitator, a_exp):
        """
        Calculates the standard Mean Squared Error (MSE) loss.

        Args:
            a_imitator (torch.Tensor): Predicted actions from the model.
            a_exp (torch.Tensor): Ground truth (expert) actions.

        Returns:
            torch.Tensor: The MSE loss value.
        """
        loss = self.loss(a_imitator, a_exp)
        return loss

    def weighted_criterion(self, a_imitator, a_exp):
        """
        Calculates a weighted MSE loss, where weights are based on the absolute value of expert actions.
        This can be useful to emphasize learning in critical (e.g., larger steering angle) situations.

        Args:
            a_imitator (torch.Tensor): Predicted actions from the model.
            a_exp (torch.Tensor): Ground truth (expert) actions.

        Returns:
            torch.Tensor: The weighted MSE loss value.
        """
        assert self.exp_factor >= 0, "exp_factor must be non-negative for weighted loss."
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor) # Weights increase exponentially with action magnitude
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def release(self, sdir):
        """
        Saves the trained model's state dictionary and optimizer's state dictionary.

        Args:
            sdir (str): Directory path to save the model files.
        """
        torch.save(self.state_dict(), sdir + "policy_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + "policy_optim.pth")

    def load(self, ldir):
        """
        Loads a trained model's state dictionary and optimizer's state dictionary.

        Args:
            ldir (str): Directory path to load the model files from.

        Returns:
            bool: True if parameters were loaded successfully, False otherwise.
        """
        try:
            print("Loading parameters to CPU...") # More descriptive message
            self.load_state_dict(
                torch.load(
                    ldir + "policy_model.pth",
                    map_location=torch.device('cpu')
                )
            )
            self.optimizer.load_state_dict(
                torch.load(
                    ldir + "policy_optim.pth",
                    map_location=torch.device('cpu')
                )
            )
            print("Parameters loaded from: " + ldir) # Consistent message format
            return True
        except Exception as e: # Catch specific exception for better debugging
            print(f"Parameters could not be loaded. Error: {e}")
            return False

    def count_params(self):
        """
        Counts the total number of learnable parameters in the network.

        Returns:
            int: Total number of learnable parameters.
        """
        self.num_params = sum(param.numel() for param in self.parameters())
        return self.num_params

    def nn_structure(self):
        """
        Returns a dictionary containing the model's direct child modules.

        Returns:
            dict: A dictionary mapping module names to their instances.
        """
        dict_layer = {i: self._modules[i] for i in self._modules.keys()}
        return dict_layer


if __name__ == "__main__":
    print("--- Starting Model Functionality Quick Check ---")

    # Define common parameters
    image_channels, image_height, image_width = (1, 480, 640) # Single image shape (C, H, W)
    s = (image_channels, image_height, image_width)
    
    output_actions_dim = 3 # Model output action dimension (e.g., steering, throttle, brake)
    a = output_actions_dim # Corresponds to Convolution_Model's 'a' parameter

    time_sequence_length = 16 # Sequence length, i.e., number of frames
    batch_size = 4            # Batch size

    # Parameters for RNN models
    hidden_size_rnn = 64      # Hidden size for GRU/LSTM
    num_units_ctgru = 64      # Number of units for CTGRU
    M_ctgru = 8               # Number of trajectories for CTGRU
    
    # 1. Prepare dummy input data
    # For Convolution_Model, input is a single image: (batch_size, channels, height, width)
    dummy_single_image_input = torch.randn(batch_size, image_channels, image_height, image_width)
    print(f"\nDummy single image input shape (C_Model): {dummy_single_image_input.shape}")

    # For RNN models with CNN_Head, input is an image sequence: (batch_size, time_sequence, channels, height, width)
    dummy_sequence_images_input = torch.randn(
        batch_size, time_sequence_length, image_channels, image_height, image_width
    )
    print(f"Dummy image sequence input shape (CNN_Head + RNN): {dummy_sequence_images_input.shape}")
    print("-" * 40)

    # 2. Initialize and test Convolution_Model (stand-alone CNN model)
    try:
        print("Testing Convolution_Model...")
        # Assuming Convolution_Model is defined or imported from your codebase
        policy1 = Convolution_Model(s, a)
        output_policy1 = policy1(dummy_single_image_input)
        print(f"  Input shape: {dummy_single_image_input.shape}")
        print(f"  Convolution_Model Output shape: {output_policy1.shape}")
        expected_shape = (batch_size, output_actions_dim)
        assert output_policy1.shape == expected_shape, f"Shape mismatch! Expected: {expected_shape}, Actual: {output_policy1.shape}"
        print("  Convolution_Model test passed!")
    except Exception as e:
        print(f"  Convolution_Model test failed! Error: {e}")
    print("-" * 40)

    # 3. Initialize and test ConvolutionHead_Nvidia (CNN Feature Extractor)
    # This CNN head's `total_features` is dynamically calculated based on its architecture
    cnn_head_nvidia = None # Initialize to None in case of failure
    try:
        print("Testing ConvolutionHead_Nvidia...")
        cnn_head_nvidia = ConvolutionHead_Nvidia(
            s,
            time_sequence=time_sequence_length,
            num_filters=32,
            features_per_filter=4 
        )
        # Verify if CNN_Head correctly exposes total_features attribute
        cnn_nvidia_features_extracted = cnn_head_nvidia.total_features if hasattr(cnn_head_nvidia, 'total_features') else None
        if cnn_nvidia_features_extracted is None:
             raise AttributeError("'total_features' attribute not found in ConvolutionHead_Nvidia. Check its definition.")
        print(f"  ConvolutionHead_Nvidia extracted features (total_features): {cnn_nvidia_features_extracted}")
        
        output_cnn_head_nvidia = cnn_head_nvidia(dummy_sequence_images_input)
        print(f"  Input shape: {dummy_sequence_images_input.shape}")
        print(f"  ConvolutionHead_Nvidia Output shape: {output_cnn_head_nvidia.shape}")
        expected_shape = (batch_size, time_sequence_length, cnn_nvidia_features_extracted)
        assert output_cnn_head_nvidia.shape == expected_shape, f"Shape mismatch! Expected: {expected_shape}, Actual: {output_cnn_head_nvidia.shape}"
        print("  ConvolutionHead_Nvidia test passed!")
    except Exception as e:
        print(f"  ConvolutionHead_Nvidia test failed! Error: {e}")
    print("-" * 40)

    # 4. Initialize and test ConvolutionHead_ResNet (CNN Feature Extractor)
    try:
        print("Testing ConvolutionHead_ResNet...")
        # Assuming ConvolutionHead_ResNet takes img_dim (s) and time_sequence
        cnn_head_resnet = ConvolutionHead_ResNet(
            s,
            time_sequence=time_sequence_length,
            # Add any other specific parameters for ResNet if its __init__ requires them
        )
        resnet_features_extracted = cnn_head_resnet.total_features if hasattr(cnn_head_resnet, 'total_features') else None
        if resnet_features_extracted is None:
             raise AttributeError("'total_features' attribute not found in ConvolutionHead_ResNet. Check its definition.")
        print(f"  ConvolutionHead_ResNet extracted features (total_features): {resnet_features_extracted}")

        output_cnn_head_resnet = cnn_head_resnet(dummy_sequence_images_input)
        print(f"  Input shape: {dummy_sequence_images_input.shape}")
        print(f"  ConvolutionHead_ResNet Output shape: {output_cnn_head_resnet.shape}")
        expected_shape_resnet = (batch_size, time_sequence_length, resnet_features_extracted)
        assert output_cnn_head_resnet.shape == expected_shape_resnet, f"Shape mismatch! Expected: {expected_shape_resnet}, Actual: {output_cnn_head_resnet.shape}"
        print("  ConvolutionHead_ResNet test passed!")
    except Exception as e:
        print(f"  ConvolutionHead_ResNet test failed! Error: {e}")
    print("-" * 40)

    # 5. Initialize and test ConvolutionHead_AlexNet (CNN Feature Extractor)
    try:
        print("Testing ConvolutionHead_AlexNet...")
        # Assuming ConvolutionHead_AlexNet takes img_dim (s) and time_sequence
        cnn_head_alexnet = ConvolutionHead_AlexNet(
            s,
            time_sequence=time_sequence_length,
            # Add any other specific parameters for AlexNet if its __init__ requires them
        )
        alexnet_features_extracted = cnn_head_alexnet.total_features if hasattr(cnn_head_alexnet, 'total_features') else None
        if alexnet_features_extracted is None:
             raise AttributeError("'total_features' attribute not found in ConvolutionHead_AlexNet. Check its definition.")
        print(f"  ConvolutionHead_AlexNet extracted features (total_features): {alexnet_features_extracted}")

        output_cnn_head_alexnet = cnn_head_alexnet(dummy_sequence_images_input)
        print(f"  Input shape: {dummy_sequence_images_input.shape}")
        print(f"  ConvolutionHead_AlexNet Output shape: {output_cnn_head_alexnet.shape}")
        expected_shape_alexnet = (batch_size, time_sequence_length, alexnet_features_extracted)
        assert output_cnn_head_alexnet.shape == expected_shape_alexnet, f"Shape mismatch! Expected: {expected_shape_alexnet}, Actual: {output_cnn_head_alexnet.shape}"
        print("  ConvolutionHead_AlexNet test passed!")
    except Exception as e:
        print(f"  ConvolutionHead_AlexNet test failed! Error: {e}")
    print("-" * 40)


    # 6. Initialize and test LSTM_Model (using ConvolutionHead_Nvidia as an example)
    # This test (and subsequent RNN tests) will only run if cnn_head_nvidia was successfully initialized.
    if cnn_head_nvidia:
        try:
            print("Testing LSTM_Model...")
            # Assuming LSTM_Model is defined or imported
            lstm_model = LSTM_Model(
                conv_head=cnn_head_nvidia, # Using the Nvidia CNN head for this test
                time_step=time_sequence_length,
                hidden_size=hidden_size_rnn,
                output=output_actions_dim
            )
            output_lstm = lstm_model(dummy_sequence_images_input)
            print(f"  Input shape: {dummy_sequence_images_input.shape}")
            print(f"  LSTM_Model Output shape: {output_lstm.shape}")
            expected_shape = (batch_size, time_sequence_length, output_actions_dim)
            assert output_lstm.shape == expected_shape, f"Shape mismatch! Expected: {expected_shape}, Actual: {output_lstm.shape}"
            print("  LSTM_Model test passed!")
        except Exception as e:
            print(f"  LSTM_Model test failed! Error: {e}")
    else:
        print("Skipping LSTM_Model test as ConvolutionHead_Nvidia failed to initialize.")
    print("-" * 40)

    # 7. Initialize and test GRU_Model (using ConvolutionHead_Nvidia as an example)
    if cnn_head_nvidia:
        try:
            print("Testing GRU_Model...")
            # Assuming GRU_Model is defined or imported
            gru_model = GRU_Model(
                conv_head=cnn_head_nvidia, # Using the Nvidia CNN head for this test
                time_step=time_sequence_length,
                hidden_size=hidden_size_rnn,
                output=output_actions_dim
            )
            output_gru = gru_model(dummy_sequence_images_input)
            print(f"  Input shape: {dummy_sequence_images_input.shape}")
            print(f"  GRU_Model Output shape: {output_gru.shape}")
            expected_shape = (batch_size, time_sequence_length, output_actions_dim)
            assert output_gru.shape == expected_shape, f"Shape mismatch! Expected: {expected_shape}, Actual: {output_gru.shape}"
            print("  GRU_Model test passed!")
        except Exception as e:
            print(f"  GRU_Model test failed! Error: {e}")
    else:
        print("Skipping GRU_Model test as ConvolutionHead_Nvidia failed to initialize.")
    print("-" * 40)

    # 8. Initialize and test CTGRU_Model (using ConvolutionHead_Nvidia as an example)
    if cnn_head_nvidia:
        try:
            print("Testing CTGRU_Model...")
            # Assuming CTGRU_Model is defined or imported
            ctgru_model = CTGRU_Model(
                num_units=num_units_ctgru,
                conv_head=cnn_head_nvidia, # Using the Nvidia CNN head for this test
                M=M_ctgru,
                time_step=time_sequence_length,
                output=output_actions_dim
            )
            output_ctgru = ctgru_model(dummy_sequence_images_input)
            print(f"  Input shape: {dummy_sequence_images_input.shape}")
            print(f"  CTGRU_Model Output shape: {output_ctgru.shape}")
            expected_shape = (batch_size, time_sequence_length, output_actions_dim)
            assert output_ctgru.shape == expected_shape, f"Shape mismatch! Expected: {expected_shape}, Actual: {output_ctgru.shape}"
            print("  CTGRU_Model test passed!")
        except Exception as e:
            print(f"  CTGRU_Model test failed! Error: {e}")
    else:
        print("Skipping CTGRU_Model test as ConvolutionHead_Nvidia failed to initialize.")
    print("-" * 40)

    print("\n--- All Model Functionality Quick Checks Completed! ---")
