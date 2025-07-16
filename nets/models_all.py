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
from nets.mlp_head import MLPHead


class BaseRNNModel(nn.Module):
    """A base class for RNN models to handle multi-modal feature fusion."""

    def __init__(self, depth_cnn_head, state_mlp_head, rgb_cnn_head=None,
                 time_step=16, output=3, use_cuda=True):
        super(BaseRNNModel, self).__init__()
        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")
        self.loss = nn.MSELoss(reduction='mean')
        self.exp_factor = 0.1

        # Store feature extractor heads
        self.depth_cnn_head = depth_cnn_head
        self.state_mlp_head = state_mlp_head
        self.rgb_cnn_head = rgb_cnn_head

        self.time_step = time_step
        self.output = output

        # Dynamically calculate the total input size for the RNN layer
        self.total_input_size = 0
        if self.depth_cnn_head:
            self.total_input_size += self.depth_cnn_head.total_features
        if self.state_mlp_head:
            self.total_input_size += self.state_mlp_head.output_dim
        if self.rgb_cnn_head:
            self.total_input_size += self.rgb_cnn_head.total_features

    def _fuse_features(self, depth_img, state_data, rgb_img=None):
        """
        Extracts and fuses features from multiple input modalities.

        Args:
            depth_img (torch.Tensor): Depth image sequence.
            state_data (torch.Tensor): State data sequence.
            rgb_img (torch.Tensor, optional): RGB image sequence. Defaults to None.

        Returns:
            torch.Tensor: A single fused feature tensor.
        """
        features_to_fuse = []

        # 1. Extract depth features
        depth_features = self.depth_cnn_head(depth_img)
        features_to_fuse.append(depth_features)

        # 2. Extract state features
        state_features = self.state_mlp_head(state_data)
        features_to_fuse.append(state_features)

        # 3. Extract RGB features if available
        if self.rgb_cnn_head is not None and rgb_img is not None:
            rgb_features = self.rgb_cnn_head(rgb_img)
            features_to_fuse.append(rgb_features)

        # 4. Concatenate all features along the feature dimension (dim=2)
        fused_features = torch.cat(features_to_fuse, dim=2)
        return fused_features

    def criterion(self, a_imitator, a_exp):
        return self.loss(a_imitator, a_exp)

    def weighted_criterion(self, a_imitator, a_exp):
        assert self.exp_factor >= 0, "exp_factor must be non-negative for weighted loss."
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor)
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def release(self, sdir):
        torch.save(self.state_dict(), sdir + "policy_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + "policy_optim.pth")

    def load(self, ldir):
        try:
            self.load_state_dict(
                torch.load(ldir + "policy_model.pth", map_location=torch.device('cpu'))
            )
            self.optimizer.load_state_dict(
                torch.load(ldir + "policy_optim.pth", map_location=torch.device('cpu'))
            )
            print("Loaded parameters from: " + ldir)
            return True
        except Exception as e:
            print(f"Parameters could not be loaded. Error: {e}")
            return False

    def count_params(self):
        return sum(param.numel() for param in self.parameters())

    def nn_structure(self):
        return {i: self._modules[i] for i in self._modules.keys()}


class GRU_Model(BaseRNNModel):
    """This class defines GRU model, layer is equal to 1."""

    def __init__(self, depth_cnn_head, state_mlp_head, rgb_cnn_head=None,
                 time_step=16, hidden_size=64, output=3, use_cuda=True):
        """
        Initializes the GRU_Model.
        """
        super(GRU_Model, self).__init__(
            depth_cnn_head, state_mlp_head, rgb_cnn_head,
            time_step, output, use_cuda
        )

        self.hidden_size = hidden_size
        self.GRU = nn.GRU(self.total_input_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, depth_img, state_data, rgb_img=None):
        """
        Defines the forward pass of the GRU model.

        Args:
            depth_img (torch.Tensor): Shape (B, T, C_d, H, W).
            state_data (torch.Tensor): Shape (B, T, F_s).
            rgb_img (torch.Tensor, optional): Shape (B, T, C_r, H, W).

        Returns:
            torch.Tensor: Output tensor with predicted actions, shape (batch_size, time_sequence, output).
        """
        batch_size = depth_img.shape[0]

        fused_features = self._fuse_features(depth_img, state_data, rgb_img)

        x_out, _ = self.GRU(fused_features)
        x_out = x_out.contiguous().view(-1, self.hidden_size)
        x_out = self.linear(x_out)
        x_out = x_out.view(batch_size, self.time_step, self.output)
        return x_out

    def evaluate_on_single_sequence(self, depth_img, state_data, rgb_img=None, hidden_state=None):
        """
        Evaluates the model on a single sequence (e.g., for test or validation).
        """
        fused_features = self._fuse_features(depth_img, state_data, rgb_img)

        if hidden_state is None:
            hidden_state = torch.zeros((1, fused_features.shape[0], self.hidden_size),
                                       device=fused_features.device)

        result, hidden_state_out = self.GRU(fused_features, hidden_state)

        result = result.contiguous().view(-1, self.hidden_size)
        result = self.linear(result)
        result = result.view(fused_features.shape[0], self.time_step, self.output)
        return result, hidden_state_out


class LSTM_Model(BaseRNNModel):
    """This class defines the LSTM model, with num_layers set to 1."""

    def __init__(self, depth_cnn_head, state_mlp_head, rgb_cnn_head=None,
                 time_step=16, hidden_size=64, output=3, use_cuda=True):
        """
        Initializes the LSTM_Model.
        """
        super(LSTM_Model, self).__init__(
            depth_cnn_head, state_mlp_head, rgb_cnn_head,
            time_step, output, use_cuda
        )

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.total_input_size,
                            self.hidden_size,
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, depth_img, state_data, rgb_img=None):
        """
        Defines the forward pass for the LSTM model.
        """
        batch_size = depth_img.shape[0]
        fused_features = self._fuse_features(depth_img, state_data, rgb_img)

        x_out, _ = self.lstm(fused_features)
        x_out = x_out.contiguous().view(-1, self.hidden_size)
        x_out = self.linear(x_out)
        x_out = x_out.view(batch_size, self.time_step, self.output)
        return x_out

    def evaluate_on_single_sequence(self, depth_img, state_data, rgb_img=None, hidden_state=None):
        """
        Evaluates the model on a single sequence sequentially (e.g., for valid or test).
        """
        fused_features = self._fuse_features(depth_img, state_data, rgb_img)

        if hidden_state is None:
            h0 = torch.zeros((1, fused_features.shape[0], self.hidden_size), device=fused_features.device)
            c0 = torch.zeros((1, fused_features.shape[0], self.hidden_size), device=fused_features.device)
            hidden_state = (h0, c0)

        result, hidden_state_out = self.lstm(fused_features, hidden_state)

        result = result.contiguous().view(-1, self.hidden_size)
        result = self.linear(result)
        result = result.view(fused_features.shape[0], self.time_step, self.output)
        return result, hidden_state_out


class CTGRU_Model(BaseRNNModel):
    """
    This class defines a Continuous-Time Gated Recurrent Unit (CT-GRU) model.
    It incorporates multiple "traces" (M) with varying time constants to capture
    information at different temporal scales.
    """

    def __init__(self, num_units, conv_head,
                 M=8, time_step=16, output=3, use_cuda=True,
                 depth_cnn_head=None, state_mlp_head=None, rgb_cnn_head=None): # Added for compatibility
        """
        Initializes the CTGRU_Model.
        """
        # This is a bit of a hack to maintain backward compatibility with old scripts
        # while moving to the new BaseRNNModel structure.
        if depth_cnn_head is None: depth_cnn_head = conv_head

        super(CTGRU_Model, self).__init__(
            depth_cnn_head, state_mlp_head, rgb_cnn_head,
            time_step, output, use_cuda
        )
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

        # Image interval, 0.04 for training, 0.2 for simulation test
        self.delta_t = torch.tensor(0.04, device=self.device)

        self.feature_number = self.total_input_size

        # Linear layers for CT-GRU gates
        self.linear_r = nn.Linear(self.feature_number + self._num_units, self._num_units * M)
        self.linear_q = nn.Linear(self.feature_number + self._num_units, self._num_units)
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

    def forward(self, depth_img, state_data, rgb_img=None):
        """
        Defines the forward pass of the CT-GRU model.
        """
        fused_features = self._fuse_features(depth_img, state_data, rgb_img)
        x = fused_features # Use fused features as input
        
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

    def evaluate_on_single_sequence(self, depth_img, state_data, rgb_img=None, hidden_state=None):
        """
        Evaluates the model on a single sequence sequentially (e.g., for validation or test).
        """
        fused_features = self._fuse_features(depth_img, state_data, rgb_img)
        x = fused_features
        
        results = []
        if hidden_state is None:
            hidden_state = torch.zeros(
                (x.shape[0], self._num_units, self.M), device=x.device
            )

        for t in range(self.time_step):
            inputs = x[:, t, :] # Shape: (batch_size, feature_number)
            hidden_state, current_step_hidden_output = self.update(inputs, hidden_state, self.delta_t)
            results.append(current_step_hidden_output)

        results_stacked = torch.stack(results, dim=1)
        results_linear_input = results_stacked.contiguous().view(-1, self._num_units)
        results_final = self.linear(results_linear_input)
        results_final = results_final.view(x.shape[0], self.time_step, self.output)
        
        return results_final, hidden_state


if __name__ == "__main__":
    print("--- Starting Model Functionality Quick Check ---")

    # Define common parameters
    depth_channels, depth_height, depth_width = (1, 480, 640) # Single depth shape (C, H, W)
    s = (depth_channels, depth_height, depth_width)
    
    # Optional RGB input shape (if available)
    image_channels, image_height, image_width = (3, 480, 640) # RGB shape (C, H, W)
    s_rgb = (image_channels, image_height, image_width)

    use_rgb = True # Set to True if RGB input is available, otherwise False
    
    output_actions_dim = 3 # Model output action dimension (e.g., steering, throttle, brake)
    a = output_actions_dim # Corresponds to Convolution_Model's 'a' parameter

    time_sequence_length = 16 # Sequence length, i.e., number of frames
    batch_size = 4            # Batch size

    # Parameters for RNN models
    hidden_size_rnn = 64      # Hidden size for GRU/LSTM
    num_units_ctgru = 64      # Number of units for CTGRU
    state_input_dim = 2       # e.g., (distance, angle)
    M_ctgru = 8               # Number of trajectories for CTGRU
    
    # Prepare dummy input data
    # For RNN models with CNN_Head, input is an image sequence: (batch_size, time_sequence, channels, height, width)
    dummy_sequence_images_input = torch.randn(
        batch_size, time_sequence_length, depth_channels, depth_height, depth_width
    )
    # For multi-modal RNNs, we also need state data
    dummy_state_data_input = torch.randn(batch_size, time_sequence_length, state_input_dim)
    
    print(f"Dummy image sequence input shape (CNN_Head + RNN): {dummy_sequence_images_input.shape}")
    print(f"Dummy state data input shape: {dummy_state_data_input.shape}")
    print("-" * 40)

    # 1. Initialize Feature Extraction Heads
    print("Initializing Feature Extraction Heads...")

    # Initialize Depth CNN Heads
    depth_head_nvidia, depth_head_resnet, depth_head_alexnet = None, None, None
    try:
        depth_head_nvidia = ConvolutionHead_Nvidia(s, time_sequence=time_sequence_length, num_filters=32, features_per_filter=4)
        print(f"  Depth CNN Head (Nvidia) initialized. Output features: {depth_head_nvidia.total_features}")
        depth_head_resnet = ConvolutionHead_ResNet(s, time_sequence=time_sequence_length, num_filters=32, features_per_filter=4)
        print(f"  Depth CNN Head (ResNet) initialized. Output features: {depth_head_resnet.total_features}")
        depth_head_alexnet = ConvolutionHead_AlexNet(s, time_sequence=time_sequence_length, num_filters=32, features_per_filter=4)
        print(f"  Depth CNN Head (AlexNet) initialized. Output features: {depth_head_alexnet.total_features}")
    except Exception as e:
        print(f"  Depth CNN Head initialization failed! Error: {e}")
        if not depth_head_nvidia:
            print("Nvidia初始化失败")
        if not depth_head_resnet:
            print("Resnet初始化失败")
        if not depth_head_alexnet:
            print("Alexnet初始化失败")

    # Initialize RGB CNN Heads (Optional)
    rgb_head_nvidia, rgb_head_resnet, rgb_head_alexnet = None, None, None
    if use_rgb:  # Only initialize if RGB input shape is defined
        try:
            rgb_head_nvidia = ConvolutionHead_Nvidia(s_rgb, time_sequence=time_sequence_length, num_filters=32, features_per_filter=4)
            print(f"  RGB CNN Head (Nvidia) initialized. Output features: {rgb_head_nvidia.total_features}")
            rgb_head_resnet = ConvolutionHead_ResNet(s_rgb, time_sequence=time_sequence_length, num_filters=32, features_per_filter=4)
            print(f"  RGB CNN Head (ResNet) initialized. Output features: {rgb_head_resnet.total_features}")
            rgb_head_alexnet = ConvolutionHead_AlexNet(s_rgb, time_sequence=time_sequence_length, num_filters=32, features_per_filter=4)
            print(f"  RGB CNN Head (AlexNet) initialized. Output features: {rgb_head_alexnet.total_features}")
        except Exception as e:
            print(f"  RGB CNN Head initialization failed! Error: {e}")
            if not rgb_head_nvidia:
                print("Nvidia初始化失败")
            if not rgb_head_resnet:
                print("Resnet初始化失败")
            if not rgb_head_alexnet:
                print("Alexnet初始化失败")
    else:
        print("  RGB input shape not defined, skipping RGB CNN Head initialization.")

    # Initialize State MLP Head
    mlp_head = None
    try:
        mlp_head = MLPHead(input_dim=state_input_dim, output_dim=64)
        print(f"  State MLP Head initialized. Output features: {mlp_head.output_dim}")
    except Exception as e:
        print(f"  State MLP Head initialization failed! Error: {e}")

    all_heads = [depth_head_nvidia, depth_head_resnet, depth_head_alexnet,
                 rgb_head_nvidia, rgb_head_resnet, rgb_head_alexnet, mlp_head]
    if not any(all_heads):  # Check if all heads are None
        print("All head initializations failed. Please check dependencies and configurations.")

    print("-" * 40)


    # 3. Test RNN Models with Fused Features
    rnn_models = {
        'LSTM_Model': LSTM_Model,
        'GRU_Model': GRU_Model,
        'CTGRU_Model': CTGRU_Model,
    }


    # Function to test RNN models with given heads
    def test_rnn_models(depth_head_arg, rgb_head_arg, head_type):
        if all(h is not None for h in [depth_head_arg, mlp_head]):  # Ensure required heads are initialized
            for rnn_name, RnnModelClass in rnn_models.items():
                print(f"\n  Testing {rnn_name} with {head_type} Heads (Depth + State)...")
                # Test case 1: Depth + State
                print("    - Scenario 1: Depth + State")
                model_ds = RnnModelClass(
                    depth_cnn_head=depth_head_arg, state_mlp_head=mlp_head,
                    time_step=time_sequence_length, hidden_size=hidden_size_rnn, output=output_actions_dim
                ) if rnn_name != 'CTGRU_Model' else CTGRU_Model(
                    num_units=num_units_ctgru, conv_head=depth_head_arg,  # conv_head is a legacy argument
                    depth_cnn_head=depth_head_arg, state_mlp_head=mlp_head,
                    time_step=time_sequence_length, output=output_actions_dim
                )
                output_ds = model_ds(dummy_sequence_images_input, dummy_state_data_input)
                assert output_ds.shape == (batch_size, time_sequence_length, output_actions_dim)
                print(f"      Input size: {model_ds.total_input_size}, Output shape: {output_ds.shape} -> OK")
                print(f"      Total params: {model_ds.count_params():,}")

            # Test RGB only if the corresponding RGB head and the general RGB input shape (s_rgb) are available:
            if use_rgb and all(h is not None for h in [rgb_head_arg, mlp_head]):
                for rnn_name, RnnModelClass in rnn_models.items():
                    print(f"\n  Testing {rnn_name} with {head_type} Heads (Depth + State + RGB)...")
                    print("    - Scenario 2: Depth + State + RGB")
                    dummy_rgb_input = torch.randn(batch_size, time_sequence_length, 3, image_height, image_width)
                    model_dsr = RnnModelClass(
                        depth_cnn_head=depth_head_arg, state_mlp_head=mlp_head, rgb_cnn_head=rgb_head_arg,
                        time_step=time_sequence_length, hidden_size=hidden_size_rnn, output=output_actions_dim
                    ) if rnn_name != 'CTGRU_Model' else CTGRU_Model(
                        num_units=num_units_ctgru, conv_head=depth_head_arg,
                        depth_cnn_head=depth_head_arg, state_mlp_head=mlp_head, rgb_cnn_head=rgb_head_arg,
                        time_step=time_sequence_length, output=output_actions_dim
                    )
                    output_dsr = model_dsr(dummy_sequence_images_input, dummy_state_data_input, dummy_rgb_input)
                    assert output_dsr.shape == (batch_size, time_sequence_length, output_actions_dim)
                    print(f"      Input size: {model_dsr.total_input_size}, Output shape: {output_dsr.shape} -> OK")
                    print(f"      Total params: {model_dsr.count_params():,}")
        else:
            print(f"  Skipping tests for {head_type} heads due to initialization failure.")

    # Run tests for each type of CNN head
    test_rnn_models(depth_head_nvidia, rgb_head_nvidia, "Nvidia")
    test_rnn_models(depth_head_resnet, rgb_head_resnet, "ResNet")
    test_rnn_models(depth_head_alexnet, rgb_head_alexnet, "AlexNet")

    print("\n--- All Model Functionality Quick Checks Completed! ---")
