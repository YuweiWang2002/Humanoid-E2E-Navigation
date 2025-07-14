"""Convolutional head before RNN.

This file defines CNN head before RNN in combination of CNN+RNN.
"""
# import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import os
import sys
# Add the parent directory to the Python path
# This assumes cnn_head.py is in the 'nets/' directory, and 'utils.py' is in its parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils import _image_standardization


# ResBlock, implement the ResBlock
class ResBlock(nn.Module):
    """This class defines the residual block."""

    def __init__(self, in_channel, out_channel, stride=1):
        """Initialize the object."""
        super(ResBlock, self).__init__()
        self.normal = nn.Sequential(
            nn.Conv2d(in_channel,
                      out_channel,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,
                      out_channel,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,
                          out_channel,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        """Define forward process of residual block."""
        out = self.normal(x) + self.shortcut(x)
        out = F.relu(out)
        return out


class ConvolutionHead_Nvidia(nn.Module):
    """This class defines Nvidia CNN head."""

    def __init__(self, img_dim, time_sequence,
                 num_filters=8, features_per_filter=4):
        """Initialize the object."""
        super(ConvolutionHead_Nvidia, self).__init__()
        # wyw: (3, 66, 200) -> (1, 480, 640)

        self.feature_layer = None
        self.filter_output = []
        # self.linear = []
        self.num_filters = num_filters
        self.features_per_filter = features_per_filter
        self.conv = nn.Sequential(  # wyw: (66, 200) → (1, 480, 640)
            nn.Conv2d(img_dim[0], 24, kernel_size=5, stride=2, bias=True),
            nn.ReLU(inplace=True),  # wyw: after (238,318)
            nn.MaxPool2d(kernel_size=2, stride=2), # after (119, 159)

            nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=True),
            nn.ReLU(inplace=True),  # after (58,77)
            nn.MaxPool2d(kernel_size=2, stride=2), # after (29, 38)

            nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=True),
            nn.ReLU(inplace=True),  # after (13,17)
            nn.MaxPool2d(kernel_size=2, stride=2), # after (6, 8)

            nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(64),   # after (4,6)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # after (2, 3)

            nn.Conv2d(64, self.num_filters,
                      kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(self.num_filters),   # after (2, 3)
            nn.ReLU(inplace=True),
        )
        # Calculate the flattened features size dynamically for the linear layers
        with torch.no_grad():
            # Create a dummy input tensor with the expected shape
            # (batch_size=1, channels, height, width)
            dummy_input = torch.zeros(1, img_dim[0], img_dim[1], img_dim[2])
            output_features = self.conv(dummy_input)
            # Flatten the output to get the size for the linear layer
            self.flattened_features_size = output_features.view(output_features.size(0), -1).size(1)

        # Dynamically create linear layers based on calculated flattened_features_size
        # The original code had 32 linear layers. Let's make it flexible.
        self.linear_layers = nn.ModuleList([
            nn.Linear(self.flattened_features_size // self.num_filters, self.features_per_filter)
            for _ in range(self.num_filters)
        ])

        self.img_channel = img_dim[0]
        self.img_height = img_dim[1]
        self.img_width = img_dim[2]
        self.time_sequence = time_sequence
        self.total_features = self.num_filters * self.features_per_filter

    def forward(self, x):
        """Define forward process of Nvidia CNN_head."""
        # Input x has shape (batch_size, time_sequence, channel, height, width)
        batch_size = x.shape[0]

        # Flatten batch and time dimensions for convolutional processing
        # Shape becomes -> (batch_size * time_sequence, channel, height, width)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)

        # Standardize image. NOTE: This helper function flattens the tensor to 2D.
        x = _image_standardization(x)

        # Reshape the tensor back to 4D for the convolutional layers
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)

        # Pass through convolutional layers
        # Shape -> (batch_size * time_sequence, num_filters, H', W')
        conv_features = self.conv(x)

        # Split the feature maps along the filter dimension.
        # This creates a list of tensors, each with shape:
        # (batch_size * time_sequence, 1, H', W')
        split_features = torch.split(conv_features, 1, dim=1)

        feature_vectors = []
        features_per_map = self.flattened_features_size // self.num_filters
        for i in range(self.num_filters):
            # Flatten the feature map for the i-th filter
            # Shape -> (batch_size * time_sequence, features_per_map)
            flattened_map = split_features[i].view(-1, features_per_map)

            # Pass through the corresponding linear layer and apply ReLU
            feature_vec = F.relu(self.linear_layers[i](flattened_map))
            feature_vectors.append(feature_vec)

        # Concatenate the resulting feature vectors from all filters
        # Shape -> (batch_size * time_sequence, total_features)
        concatenated_features = torch.cat(feature_vectors, dim=1)

        # Reshape to re-introduce the time sequence dimension
        # Shape -> (batch_size, time_sequence, total_features)
        feature_layer = concatenated_features.view(
            batch_size, self.time_sequence, self.total_features)

        return feature_layer

    def count_params(self):
        """Return back how many params CNN_head have."""
        return sum(param.numel() for param in self.parameters())


class ConvolutionHead_ResNet(nn.Module):
    """This class defines ResNet CNN head."""

    # use ResNet18 structure, with less channels.
    def __init__(self, img_dim, time_sequence,
                 num_filters=8, features_per_filter=4):
        """Initialize the object."""
        super(ConvolutionHead_ResNet, self).__init__()
        # wyw: (3, 66, 200) → (1, 480, 640)

        self.feature_layer = None
        self.filter_output = []
        # self.linear = []
        self.num_filters = num_filters
        self.features_per_filter = features_per_filter

        self.in_channel = 24
        # layer before Residual Block input image (66, 200)
        # wyw: (66, 200) → (1, 480, 640)
        self.conv1 = nn.Sequential( # Use img_dim[0] for input channels
            nn.Conv2d(img_dim[0], 24, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),  # (64,198)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # After (24, 239, 319) -> (24, 119, 159)
        )
        self.layer1 = self.make_layer(ResBlock, 36, 2, stride=2) # Input (24, 119, 159) -> Output (36, 60, 80) approx
        self.layer2 = self.make_layer(ResBlock, 48, 2, stride=2) # Input (36, 60, 80) -> Output (48, 30, 40) approx
        self.layer3 = self.make_layer(ResBlock, 64, 2, stride=2) # Input (48, 30, 40) -> Output (64, 15, 20) approx
        self.layer4 = self.make_layer(ResBlock, 64, 2, stride=2) # Input (64, 15, 20) -> Output (64, 8, 10) approx
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,
                      self.num_filters,
                      kernel_size=3,
                      stride=1,
                      bias=False),   # Output (num_filters, 6, 8)
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # After (num_filters, 6, 8) -> (num_filters, 3, 4)
        )

        # Calculate the flattened features size dynamically for the linear layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, img_dim[0], img_dim[1], img_dim[2])
            x = self.conv1(dummy_input)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            output_features = self.conv2(x)
            self.flattened_features_size = output_features.view(output_features.size(0), -1).size(1)


        self.linear_layers = nn.ModuleList([
            nn.Linear(self.flattened_features_size // self.num_filters, self.features_per_filter)
            for _ in range(self.num_filters)
        ])

        self.img_channel = img_dim[0]
        self.img_height = img_dim[1]
        self.img_width = img_dim[2]
        self.time_sequence = time_sequence
        self.total_features = self.num_filters * self.features_per_filter

    def make_layer(self, block, channels, num_blocks, stride):
        """Make layers of resblock."""
        strides = [stride] + [1]*(num_blocks-1)
        # create a list [stride,1,1,..,1], the number is:num_blocks-1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        """Define forward process of ResNet CNN head."""
        # x has the shape (batch size, time Sequence, channel, height, width)
        # flatten the time_sequence*batch_size

        # necessary because the last batch's size is not equal
        # to set batch size
        batch_size = x.shape[0]

        # flatten x (batch_size * time_Sequence)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)

        # do pic whitening  (pic-mean)/std
        x = _image_standardization(x)

        x = x.view(-1, self.img_channel, self.img_height, self.img_width)
        x = self.conv1(x)  # shape (sample_numbers, channels, height, width)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)

        # after get the result from the conv layer,
        # split the output of each channel,
        # and feed the channels input into
        # the linear layer individually
        # slice one by one , (sample_numbers,1,height, width)
        self.filter_output = list(torch.split(x, 1, dim=1))

        feature_layer_list = []
        for i in range(self.num_filters):
            # print(filter_output[i].shape)
            # (sample_numbers, height, width)
            self.filter_output[i] = torch.squeeze(
                self.filter_output[i], dim=1)

            self.filter_output[i] = self.filter_output[i].view(-1, self.flattened_features_size // self.num_filters)

            # the output of each filter feed into linear layer
            feats = F.relu(self.linear_layers[i](self.filter_output[i]))
            feature_layer_list.append(feats)

        # concat the features from each filter together
        self.feature_layer = torch.cat(feature_layer_list, 1)
        feature_layer = self.feature_layer.view(
            batch_size, self.time_sequence, self.total_features)

        return feature_layer  # (time_Sequence, batch_size, total_features)

    def count_params(self):
        """Return back how many params CNN_head have."""
        return sum(param.numel() for param in self.parameters())


class ConvolutionHead_AlexNet(nn.Module):
    """This class defines AlexNet CNN head."""

    def __init__(self, img_dim, time_sequence,
                 num_filters=8, features_per_filter=4):
        """Initialize the object."""
        super(ConvolutionHead_AlexNet, self).__init__()
        # wyw: (3, 66, 200) → (1, 480, 640)
        self.feature_layer = None
        self.filter_output = []
        # self.linear = []
        self.num_filters = num_filters
        self.features_per_filter = features_per_filter
        self.conv = nn.Sequential(  # wyw: (66, 200) → (1, 480, 640)
            nn.Conv2d(img_dim[0], 24, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # After (240, 320)

            nn.Conv2d(24, 36, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # After (120, 160)

            nn.Conv2d(36, 48, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # After (60, 80)

            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # After (30, 40)

            nn.Conv2d(64, self.num_filters,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # After (15, 20)
        )
        # Calculate the flattened features size dynamically for the linear layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, img_dim[0], img_dim[1], img_dim[2])
            output_features = self.conv(dummy_input)
            self.flattened_features_size = output_features.view(output_features.size(0), -1).size(1)
            # For num_filters=8, final size is (8, 15, 20) -> 8 * 15 * 20 = 2400
            # So, in_features for each linear layer will be 2400 / 8 = 300

        self.linear_layers = nn.ModuleList([
            nn.Linear(self.flattened_features_size // self.num_filters, self.features_per_filter)
            for _ in range(self.num_filters)
        ])

        self.img_channel = img_dim[0]
        self.img_height = img_dim[1]
        self.img_width = img_dim[2]
        self.time_sequence = time_sequence
        self.total_features = self.num_filters * self.features_per_filter

    def forward(self, x):
        """Define forward process of AlexNet CNN head."""
        # x has the shape (batch size, time Sequence, channel, height, width)
        # flatten the time_sequence*batch_size
        # necessary because the last batch's size
        # is not equal to set batch size
        batch_size = x.shape[0]

        # flatten x (batch_size * time_Sequence)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)

        # do pic whitening  (pic-mean)/std
        x = _image_standardization(x)

        x = x.view(-1, self.img_channel, self.img_height, self.img_width)
        x = self.conv(x)  # shape (sample_numbers, channels, height, width)

        # after get the result from the conv layer,
        # split the output of each channel,
        # and feed the channels input into the linear layer individually
        # slice one by one , (sample_numbers,1,height, width)
        self.filter_output = list(torch.split(x, 1, dim=1))

        feature_layer_list = []
        for i in range(self.num_filters):
            # print(filter_output[i].shape)
            self.filter_output[i] = torch.squeeze(
                self.filter_output[i], dim=1)
            # (sample_numbers, height, width)

            # flatten the output of each filter
            self.filter_output[i] = self.filter_output[i].view(-1, self.flattened_features_size // self.num_filters)

            # the output of each filter feed into linear layer
            feats = F.relu(self.linear_layers[i](self.filter_output[i]))
            feature_layer_list.append(feats)

        # concat the features from each filter together
        self.feature_layer = torch.cat(feature_layer_list, 1)

        feature_layer = self.feature_layer.view(
            batch_size, self.time_sequence, self.total_features)

        return feature_layer  # (time_Sequence, batch_size, total_features)

    def count_params(self):
        """Return back how many params CNN_head have."""
        return sum(param.numel() for param in self.parameters())


if __name__ == "__main__":
    # Test with 3 channels and 480x640 resolution
    input_dim_test = (1, 480, 640)
    # input_dim_test = (3, 480, 640)
    time_seq_test = 16
    num_filters_test = 32
    features_per_filter_test = 4

    print("--- Testing ConvolutionHead_Nvidia ---")
    nvidia_model = ConvolutionHead_Nvidia(
        input_dim_test,
        time_seq_test,
        num_filters=num_filters_test,
        features_per_filter=features_per_filter_test
    )
    # Check the calculated in_features for the linear layers
    print(f"Nvidia Model - Calculated in_features per linear layer: {nvidia_model.linear_layers[0].in_features}")
    # Expected: (32 * 2 * 3) / 32 = 6
    dummy_input_nvidia = torch.randn(2, time_seq_test, input_dim_test[0], input_dim_test[1], input_dim_test[2])
    output_nvidia = nvidia_model(dummy_input_nvidia)
    print(f"Nvidia Model output shape: {output_nvidia.shape}")
    print(f"Nvidia Model params: {nvidia_model.count_params()}\n")


    print("--- Testing ConvolutionHead_ResNet ---")
    resnet_model = ConvolutionHead_ResNet(
        input_dim_test,
        time_seq_test, # Changed time_sequence to 1 for simpler testing, adjust as needed
        num_filters=num_filters_test,
        features_per_filter=features_per_filter_test
    )
    print(f"ResNet Model - Calculated in_features per linear layer: {resnet_model.linear_layers[0].in_features}")
    # Expected: (32 * 3 * 4) / 32 = 12
    dummy_input_resnet = torch.randn(2, time_seq_test, input_dim_test[0], input_dim_test[1], input_dim_test[2])
    output_resnet = resnet_model(dummy_input_resnet)
    print(f"ResNet Model output shape: {output_resnet.shape}")
    print(f"ResNet Model params: {resnet_model.count_params()}\n")


    print("--- Testing ConvolutionHead_AlexNet ---")
    alexnet_model = ConvolutionHead_AlexNet(
        input_dim_test,
        time_seq_test, # Changed time_sequence to 1 for simpler testing, adjust as needed
        num_filters=num_filters_test,
        features_per_filter=features_per_filter_test
    )
    print(f"AlexNet Model - Calculated in_features per linear layer: {alexnet_model.linear_layers[0].in_features}")
    # Expected: (32 * 15 * 20) / 32 = 300
    dummy_input_alexnet = torch.randn(2, time_seq_test, input_dim_test[0], input_dim_test[1], input_dim_test[2])
    output_alexnet = alexnet_model(dummy_input_alexnet)
    print(f"AlexNet Model output shape: {output_alexnet.shape}")
    print(f"AlexNet Model params: {alexnet_model.count_params()}\n")
