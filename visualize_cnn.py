"""
This script visualizes the CNN's attention using Grad-CAM to understand
what parts of an image the model is focusing on.
"""

import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms

from nets.models_all import GRU_Model, LSTM_Model, CTGRU_Model
from nets.cnn_head import ConvolutionHead_Nvidia, ConvolutionHead_ResNet, ConvolutionHead_AlexNet
from nets.mlp_head import MLPHead
from dataset_all import HumanoidNavDataset
from normalization import NormalizationManager
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import NonTensorOutputTarget

def main():
    parser = argparse.ArgumentParser(description="Visualize CNN attention using Grad-CAM.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config file for model architecture.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the single depth image to visualize.")
    parser.add_argument("--output_path", type=str, default="grad_cam_output.jpg", help="Path to save the output visualization.")

    args = parser.parse_args()

    # --- Load Config and Build Model ---
    from train_all_models import load_config
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the same model as used in training
    depth_img_dim, state_dim = tuple(cfg['data']['depth_img_shape']), cfg['data']['state_dim']
    cnn_head_map = {'Nvidia': ConvolutionHead_Nvidia, 'ResNet': ConvolutionHead_ResNet, 'AlexNet': ConvolutionHead_AlexNet}
    CNN_HEAD_CLASS = cnn_head_map[cfg['model']['heads']['depth_cnn_head']]
    depth_cnn_head = CNN_HEAD_CLASS(img_dim=depth_img_dim, time_sequence=1) # Visualize single frame
    state_mlp_head = MLPHead(input_dim=state_dim, output_dim=cfg['model']['heads']['mlp']['output_dim'])
    
    rnn_type = cfg['model']['rnn_type']
    if rnn_type == 'GRU':
        model = GRU_Model(depth_cnn_head, state_mlp_head, None, time_step=1, hidden_size=cfg['model']['rnn_params']['hidden_size'], output=cfg['model']['output_dim'])
    else: # Add LSTM/CTGRU if needed
        raise NotImplementedError(f"Visualization for {rnn_type} not implemented yet.")

    # Load the trained weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # --- Prepare Input Data ---
    # Load and transform the image
    from PIL import Image
    depth_norm_manager = NormalizationManager(stats_dim=1, file_path='depth_normalization.json')
    img_transforms = transforms.Compose([transforms.ToTensor(), depth_norm_manager.get_transform()])
    
    img = Image.open(args.image_path).convert('L')
    input_tensor = img_transforms(img).unsqueeze(0).unsqueeze(0) # Add batch and time dimensions (B, T, C, H, W)
    input_tensor = input_tensor.to(device)

    # Create a dummy state tensor (since we only care about the CNN part)
    dummy_state = torch.zeros(1, 1, state_dim).to(device)

    # --- Setup Grad-CAM ---
    # The target layer is the last convolutional layer of your CNN head
    target_layer = model.depth_cnn_head.conv5
    
    # Grad-CAM works by getting the gradient of an output with respect to the target layer.
    # Since our model outputs a tensor of velocities, not a class score, we need a custom target.
    # We'll make it look at the angular velocity (vel_yaw), which is the last output.
    class VelYawTarget(NonTensorOutputTarget):
        def __init__(self, yaw_index):
            self.yaw_index = yaw_index
        def __call__(self, model_output):
            return model_output[:, :, self.yaw_index]

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [VelYawTarget(2)] # Index 2 for vel_yaw

    # Generate the CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, extra_inputs=dummy_state)
    grayscale_cam = grayscale_cam[0, :] # Take the first (and only) image

    # --- Visualize and Save ---
    # Convert single channel depth image to 3-channel for color overlay
    img_np = np.array(img) / 255.0
    rgb_img = np.stack([img_np]*3, axis=-1)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Save the image
    cv2.imwrite(args.output_path, visualization)
    print(f"Grad-CAM visualization saved to {args.output_path}")
    print("Heatmap shows regions that most influenced the robot's turning decision (vel_yaw).")

if __name__ == '__main__':
    main()
