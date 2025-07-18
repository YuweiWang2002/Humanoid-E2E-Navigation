"""
Main training script for the Humanoid Navigation task.

This script trains a multi-modal model (CNN+MLP+RNN) using preprocessed
data to predict robot velocities. It is driven by a YAML configuration file.
"""

import argparse
import time
import numpy as np
import os
import json
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml
import shutil

from nets.cnn_head import ConvolutionHead_Nvidia, ConvolutionHead_ResNet, ConvolutionHead_AlexNet
from nets.mlp_head import MLPHead
from nets.models_all import GRU_Model, LSTM_Model, CTGRU_Model
from dataset_all import HumanoidNavDataset
from early_stopping import EarlyStopping
from utils import make_dirs, save_result
from normalization import NormalizationManager

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_normalization_stats(depth_manager, state_manager, dataloader, device):
    """Iterates through the dataloader to update normalization stats for all modalities."""
    print("Updating normalization statistics for depth images and state vectors...")
    for batch in tqdm(dataloader, desc="Updating Stats"):
        # Update depth image stats (single channel)
        images = batch['depth_img'].to(device)
        # Reshape to (N, C) where N is number of pixels
        reshaped_images = images.view(-1, 1)
        depth_manager.update(reshaped_images)

        # Update state vector stats
        states = batch['state'].to(device)
        # Reshape from (B, T, D) to (B*T, D)
        reshaped_states = states.view(-1, states.shape[-1])
        state_manager.update(reshaped_states)

    depth_manager.save()
    state_manager.save()

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Train a Humanoid E2E Navigation model using a configuration file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/base_config.yml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--update_stats', action='store_true',
                        help='If set, recalculate and update ALL normalization stats before training.')

    args = parser.parse_args()
    
    cfg = load_config(args.config)

    device = torch.device("cuda" if cfg['use_cuda'] and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed_value = cfg['seed']
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)

    SDIR = os.path.join(cfg['logging']['results_dir'], cfg['run_name'])
    make_dirs(SDIR)
    shutil.copy(args.config, os.path.join(SDIR, 'config.yml'))

    if cfg['logging']['use_wandb']:
        wandb.init(project=cfg['logging']['wandb_project_name'], name=cfg['run_name'], config=cfg)

    processed_dir = cfg['data']['processed_data_dir']
    all_csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {processed_dir}")

    np.random.shuffle(all_csv_files)
    split_idx = int((1 - cfg['training']['validation_split']) * len(all_csv_files))
    train_files, valid_files = all_csv_files[:split_idx], all_csv_files[split_idx:]

    # --- Normalization ---
    depth_norm_manager = NormalizationManager(stats_dim=1, file_path='depth_normalization.json')
    state_norm_manager = NormalizationManager(stats_dim=cfg['data']['state_dim'], file_path='state_normalization.json')

    if args.update_stats:
        stats_dataset = HumanoidNavDataset(
            processed_data_dir=processed_dir, csv_files=train_files,
            sequence_length=cfg['data']['time_step'],
            transform=transforms.Compose([transforms.ToTensor()]) # No normalization yet
        )
        stats_loader = DataLoader(stats_dataset, batch_size=cfg['training']['batch_size'], num_workers=cfg['training']['num_workers'])
        update_normalization_stats(depth_norm_manager, state_norm_manager, stats_loader, device)
    
    img_transforms = transforms.Compose([transforms.ToTensor(), depth_norm_manager.get_transform()])

    train_dataset = HumanoidNavDataset(
        processed_data_dir=processed_dir, csv_files=train_files,
        sequence_length=cfg['data']['time_step'], transform=img_transforms, 
        state_mean=state_norm_manager.mean, state_std=state_norm_manager.std
    )
    valid_dataset = HumanoidNavDataset(
        processed_data_dir=processed_dir, csv_files=valid_files,
        sequence_length=cfg['data']['time_step'], transform=img_transforms, 
        state_mean=state_norm_manager.mean, state_std=state_norm_manager.std
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training']['num_workers'], pin_memory=True)

    # --- Model, Optimizer, and Loss ---
    depth_img_dim, rgb_img_dim = tuple(cfg['data']['depth_img_shape']), tuple(cfg['data']['rgb_img_shape'])
    state_dim, output_dim = cfg['data']['state_dim'], cfg['model']['output_dim']

    cnn_head_map = {'Nvidia': ConvolutionHead_Nvidia, 'ResNet': ConvolutionHead_ResNet, 'AlexNet': ConvolutionHead_AlexNet}
    CNN_HEAD_CLASS = cnn_head_map[cfg['model']['heads']['depth_cnn_head']]

    depth_cnn_head = CNN_HEAD_CLASS(img_dim=depth_img_dim, time_sequence=cfg['data']['time_step'])
    rgb_cnn_head = CNN_HEAD_CLASS(img_dim=rgb_img_dim, time_sequence=cfg['data']['time_step']) if cfg['data']['use_rgb'] else None
    state_mlp_head = MLPHead(input_dim=state_dim, output_dim=cfg['model']['heads']['mlp']['output_dim'])

    rnn_type = cfg['model']['rnn_type']
    if rnn_type == 'GRU':
        POLICY = GRU_Model(depth_cnn_head, state_mlp_head, rgb_cnn_head, time_step=cfg['data']['time_step'], hidden_size=cfg['model']['rnn_params']['hidden_size'], output=output_dim)
    elif rnn_type == 'LSTM':
        POLICY = LSTM_Model(depth_cnn_head, state_mlp_head, rgb_cnn_head, time_step=cfg['data']['time_step'], hidden_size=cfg['model']['rnn_params']['hidden_size'], output=output_dim)
    else: # CTGRU
        POLICY = CTGRU_Model(depth_cnn_head, state_mlp_head, rgb_cnn_head, time_step=cfg['data']['time_step'], output=output_dim, **cfg['model']['rnn_params']['ctgru'])

    POLICY.to(device)
    optimizer = optim.Adam(POLICY.parameters(), lr=cfg['training']['learning_rate'])
    criterion = torch.nn.MSELoss()
    
    stopper = None
    if cfg['training']['early_stopping']['enabled']:
        es_params = {k: v for k, v in cfg['training']['early_stopping'].items() if k != 'enabled'}
        stopper = EarlyStopping(**es_params)

    # --- Training Loop ---
    print("Start learning policy!")
    start_epoch = 1
    best_valid_loss = float('inf')
    train_loss_policy_o, valid_loss_policy_o = [], []

    if cfg['training']['resume']:
        checkpoint_path = os.path.join(SDIR, 'latest_checkpoint.pth')
        if os.path.isfile(checkpoint_path):
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            POLICY.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_loss_policy_o, valid_loss_policy_o = checkpoint['train_loss'], checkpoint['valid_loss']
            best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
            if stopper: stopper.best_loss = best_valid_loss
            print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg['training']['epochs'] + 1):
        POLICY.train()
        running_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch}/{cfg['training']['epochs']}", leave=False)
        for batch in train_loop:
            depth_img = batch['depth_img'].to(device)
            state_data = batch['state'].to(device)
            labels = batch['label'].to(device)
            rgb_img = batch['rgb_img'].to(device) if cfg['data']['use_rgb'] and batch['rgb_img'].numel() > 0 else None

            optimizer.zero_grad()
            outputs = POLICY(depth_img, state_data, rgb_img)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if cfg['training']['gradient_clipping']['enabled']:
                torch.nn.utils.clip_grad_norm_(POLICY.parameters(), max_norm=cfg['training']['gradient_clipping']['max_norm'])
            
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_policy_o.append(avg_train_loss)
        if cfg['logging']['use_wandb']: wandb.log({"train_loss": avg_train_loss}, step=epoch)

        # Validation loop
        POLICY.eval()
        running_valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                depth_img = batch['depth_img'].to(device)
                state_data = batch['state'].to(device)
                labels = batch['label'].to(device)
                rgb_img = batch['rgb_img'].to(device) if cfg['data']['use_rgb'] and batch['rgb_img'].numel() > 0 else None
                outputs = POLICY(depth_img, state_data, rgb_img)
                loss = criterion(outputs, labels)
                running_valid_loss += loss.item()

        avg_valid_loss = running_valid_loss / len(valid_loader)
        valid_loss_policy_o.append(avg_valid_loss)
        if cfg['logging']['use_wandb']: wandb.log({"valid_loss": avg_valid_loss}, step=epoch)
        print(f"Epoch [{epoch}/{cfg['training']['epochs']}], Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")

        is_best = avg_valid_loss < best_valid_loss
        best_valid_loss = min(avg_valid_loss, best_valid_loss)

        checkpoint_data = {
            'epoch': epoch, 'model_state_dict': POLICY.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_loss_policy_o,
            'valid_loss': valid_loss_policy_o, 'best_valid_loss': best_valid_loss
        }
        torch.save(checkpoint_data, os.path.join(SDIR, 'latest_checkpoint.pth'))
        if is_best:
            print(f"  New best model found! Saving with validation loss: {avg_valid_loss:.6f}")
            torch.save(checkpoint_data, os.path.join(SDIR, 'best_checkpoint.pth'))

        if stopper and stopper(avg_valid_loss):
            print("Early Stopping to avoid Overfitting!")
            break

    print("Training finished.")

if __name__ == '__main__':
    main()
