"""
Main training script for the Humanoid Navigation task.

This script trains a multi-modal model (CNN+MLP+RNN) using preprocessed
data to predict robot velocities.
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
from nets.cnn_head import ConvolutionHead_Nvidia, ConvolutionHead_ResNet, ConvolutionHead_AlexNet
from nets.mlp_head import MLPHead
from nets.models_all import GRU_Model, LSTM_Model, CTGRU_Model
from dataset_all import HumanoidNavDataset
from early_stopping import EarlyStopping
from utils import make_dirs, save_result

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="arg parser",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Arguments ---
    parser.add_argument("--processed_dir", type=str, default='data/processed', help="Directory with processed CSV files.")
    parser.add_argument("--results_dir", type=str, default='results', help="Directory to save training results.")
    parser.add_argument("--name", type=str, default='Humanoid_GRU_Depth_State', help="Name for the training run folder.")
    parser.add_argument("--network", type=str, default='GRU', choices=['GRU', 'LSTM', 'CTGRU'], help="Type of RNN to use.")
    parser.add_argument("--cnn_head", type=str, default='Nvidia', choices=['Nvidia', 'ResNet', 'AlexNet'], help="Type of CNN head to use.")
    parser.add_argument("--use_rgb", action='store_true', help="Flag to include RGB images in training.")
    parser.add_argument("--resume", action='store_true', help="Flag to resume training from the latest checkpoint.")

    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--sequence", type=int, default=16, help="Sequence length for RNNs.")
    parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units in RNN.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed for reproducibility.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--valid_split", type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader. Set to 0 for Windows compatibility.")
    parser.add_argument("--eval_interval", type=float, default=0.1, help="Evaluate every this fraction of training.")  # Add eval_interval

    args = parser.parse_args()

    # --- Helper function to calculate mean and std ---
    def get_dataset_stats(dataloader, device):
        """
        Calculates the mean and standard deviation of a dataset.
        """
        print("Calculating dataset statistics (mean and std)...")
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        
        # Use a temporary dataloader to iterate through all images
        for batch in tqdm(dataloader, desc="Calculating Stats"):
            # Images are shape (B, T, C, H, W)
            # We need to calculate stats over all pixels of all images in the sequence
            images = batch['depth_img'].to(device) # Change to 'rgb_img' if needed
            # Reshape to (B * T * C, H, W) to make it easier
            images = images.view(-1, images.shape[-2], images.shape[-1])

            channels_sum += torch.mean(images, dim=[0, 1, 2])
            channels_squared_sum += torch.mean(images**2, dim=[0, 1, 2])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
        return mean, std

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed_value = args.seed
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)

    SDIR = os.path.join(args.results_dir, args.name)
    make_dirs(SDIR)

    # --- Initialize wandb ---
    wandb.init(project="Humanoid-E2E-Navigation", name=args.name, config=args)
    wandb.config.update(args)  # Log arguments to wandb

    print(f"Initialized wandb run: {wandb.run.name}")


    # --- Data Loading ---
    # --- Correct Data Splitting (by Trajectory) ---
    # 1. Get a list of all trajectory files
    all_csv_files = [f for f in os.listdir(args.processed_dir) if f.endswith('.csv')]
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {args.processed_dir}")

    # 2. Shuffle the list of files for random splitting
    np.random.shuffle(all_csv_files)

    # 3. Split the list of files into training and validation sets
    split_idx = int((1 - args.valid_split) * len(all_csv_files))
    train_files = all_csv_files[:split_idx]
    valid_files = all_csv_files[split_idx:]

    # 4. Create a temporary dataset to calculate stats from training data ONLY
    # We only need ToTensor to convert images for calculation
    stats_dataset = HumanoidNavDataset(
        processed_data_dir=args.processed_dir,
        csv_files=train_files, # Use ONLY training files for stats
        sequence_length=args.sequence,
        use_rgb=args.use_rgb,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    stats_loader = DataLoader(stats_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    
    # Calculate mean and std
    # Note: This can take a while on the first run. For subsequent runs, you could
    # save these values to a file and load them to save time.
    d_mean, d_std = get_dataset_stats(stats_loader, device)
    print(f"Calculated Depth Mean: {d_mean.cpu().numpy()}, Std: {d_std.cpu().numpy()}")

    # --- Sanity Check for Std ---
    # If std is 0, it means all images are identical, causing a division-by-zero error.
    # We add a small epsilon to prevent a crash, but this indicates a data problem.
    if torch.any(d_std == 0):
        print("\n" + "="*50)
        print("!!! WARNING: Calculated Standard Deviation is ZERO. !!!")
        print("This strongly suggests that all your input images are identical (e.g., all black).")
        print("Adding a small value (epsilon) to std to prevent a crash.")
        print("THE UNDERLYING DATA ISSUE SHOULD BE FIXED for meaningful training.")
        print("="*50 + "\n")
        d_std = d_std + 1e-6  # Add a small epsilon

    # Now define the final transforms with the calculated stats
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=d_mean, std=d_std)
    ])

    # 4. Create two separate Dataset instances
    train_dataset = HumanoidNavDataset(
        processed_data_dir=args.processed_dir,
        csv_files=train_files,
        sequence_length=args.sequence,
        use_rgb=args.use_rgb,
        transform=img_transforms # Apply full transforms
    )

    valid_dataset = HumanoidNavDataset(
        processed_data_dir=args.processed_dir,
        csv_files=valid_files,
        sequence_length=args.sequence,
        use_rgb=args.use_rgb,
        transform=img_transforms # Apply the same transforms to validation set
    )

    print(f"Dataset split: {len(train_dataset)} training samples, {len(valid_dataset)} validation samples.")

    # --- Debugging Step ---
    # On Windows, num_workers must be 0 to avoid multiprocessing errors with some torchvision versions.
    print(f"DEBUG: DataLoader num_workers is set to: {args.num_workers}")
    assert args.num_workers == 0, "On Windows, num_workers must be 0. Please check your arguments."

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Model, Optimizer, and Loss ---
    # Image and state dimensions
    depth_img_dim = (1, 480, 640)
    rgb_img_dim = (3, 480, 640)
    state_dim = 2
    output_dim = 3

    # 1. Create Feature Extractor Heads
    cnn_head_map = {
        'Nvidia': ConvolutionHead_Nvidia,
        'ResNet': ConvolutionHead_ResNet,
        'AlexNet': ConvolutionHead_AlexNet
    }
    CNN_HEAD_CLASS = cnn_head_map[args.cnn_head]

    depth_cnn_head = CNN_HEAD_CLASS(img_dim=depth_img_dim, time_sequence=args.sequence)
    rgb_cnn_head = CNN_HEAD_CLASS(img_dim=rgb_img_dim, time_sequence=args.sequence) if args.use_rgb else None
    state_mlp_head = MLPHead(input_dim=state_dim)

    # 2. Create the main RNN Model
    if args.network == 'GRU':
        POLICY = GRU_Model(
            depth_cnn_head=depth_cnn_head,
            state_mlp_head=state_mlp_head,
            rgb_cnn_head=rgb_cnn_head,
            time_step=args.sequence,
            hidden_size=args.hidden,
            output=output_dim
        )
    elif args.network == 'LSTM':
        POLICY = LSTM_Model(
            depth_cnn_head=depth_cnn_head,
            state_mlp_head=state_mlp_head,
            rgb_cnn_head=rgb_cnn_head,
            time_step=args.sequence,
            hidden_size=args.hidden,
            output=output_dim
        )
    elif args.network == 'CTGRU':
        POLICY = CTGRU_Model(
            depth_cnn_head=depth_cnn_head,
            state_mlp_head=state_mlp_head,
            rgb_cnn_head=rgb_cnn_head,
            time_step=args.sequence,
            hidden_size=args.hidden,
            output=output_dim
        )
    else:
        raise ValueError(f"Unknown network type: {args.network}")


    POLICY.to(device)
    print(f"Model: {args.network} with {args.cnn_head} head. Total parameters: {POLICY.count_params():,}")

    optimizer = optim.Adam(POLICY.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    stopper = EarlyStopping(length=10)

    # --- Training Loop ---
    start_epoch = 1
    best_valid_loss = float('inf')
    train_loss_policy_o = []
    valid_loss_policy_o = []

    # --- Load Checkpoint if resuming ---
    if args.resume:
        checkpoint_path = os.path.join(SDIR, 'latest_checkpoint.pth')
        if os.path.isfile(checkpoint_path):
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            POLICY.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_loss_policy_o = checkpoint['train_loss']
            valid_loss_policy_o = checkpoint['valid_loss']
            best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
            stopper.best_loss = best_valid_loss # Sync stopper
            print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"=> no checkpoint found at '{checkpoint_path}', starting from scratch.")


    print("Start learning policy!")
    for epoch in range(start_epoch, args.epoch + 1):
        # Training
        POLICY.train()
        running_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch}/{args.epoch}", leave=False)
        for batch in train_loop:
            depth_img = batch['depth_img'].to(device)
            state_data = batch['state'].to(device)
            labels = batch['label'].to(device)
            rgb_img = batch['rgb_img'].to(device) if args.use_rgb and batch['rgb_img'].numel() > 0 else None

            optimizer.zero_grad()

            # Forward pass
            outputs = POLICY(depth_img, state_data, rgb_img)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_policy_o.append(avg_train_loss)
        wandb.log({"train_loss": avg_train_loss}, step=epoch)

        # Validation
        POLICY.eval()
        running_valid_loss = 0.0
        valid_loop = tqdm(valid_loader, desc="Validating", leave=False)
        with torch.no_grad():
            num_samples = 0
            total_loss = 0
            # Initialize lists to store individual loss components
            losses = []

            for batch in valid_loop:
                depth_img = batch['depth_img'].to(device)
                state_data = batch['state'].to(device)
                labels = batch['label'].to(device)
                rgb_img = batch['rgb_img'].to(device) if args.use_rgb and batch['rgb_img'].numel() > 0 else None

                outputs = POLICY(depth_img, state_data, rgb_img)
                loss = criterion(outputs, labels)
                # Store the total number of samples and accumulate the loss
                num_samples += labels.size(0)  # Accumulate the number of samples
                total_loss += loss.item() * labels.size(0)  # Multiply loss by batch size
                losses.append(loss.item())
                running_valid_loss += loss.item()  # This line seems unnecessary now but kept for consistency
                valid_loop.set_postfix(loss=loss.item())

            # Calculate the average loss
            avg_loss = total_loss / num_samples
            print(f"Avg Loss: {avg_loss:.4f}")

        avg_valid_loss = running_valid_loss / len(valid_loader)
        valid_loss_policy_o.append(avg_valid_loss)
        wandb.log({"valid_loss": avg_valid_loss}, step=epoch)
        print(f"Epoch [{epoch}/{args.epoch}], Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")

        # --- Save Checkpoint ---
        is_best = avg_valid_loss < best_valid_loss
        best_valid_loss = min(avg_valid_loss, best_valid_loss)

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': POLICY.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss_policy_o,
            'valid_loss': valid_loss_policy_o,
            'best_valid_loss': best_valid_loss
        }

        # Save the latest checkpoint
        torch.save(checkpoint_data, os.path.join(SDIR, 'latest_checkpoint.pth'))

        if is_best:
            print(f"  New best model found! Saving 'best_checkpoint.pth' with validation loss: {avg_valid_loss:.6f}")
            torch.save(checkpoint_data, os.path.join(SDIR, 'best_checkpoint.pth'))

        # Early stopping
        if stopper(avg_valid_loss):
            print("Early Stopping to avoid Overfitting!")
            break

        # --- Periodic Evaluation ---
        if epoch % max(1, int(args.epoch * args.eval_interval)) == 0 or epoch == args.epoch:
            POLICY.eval()
            eval_loss = 0.0
            eval_steps = 0
            with torch.no_grad():
                for batch in valid_loader:  # Re-use valid_loader for evaluation
                    depth_img = batch['depth_img'].to(device)
                    state_data = batch['state'].to(device)
                    labels = batch['label'].to(device)
                    rgb_img = batch['rgb_img'].to(device) if args.use_rgb and batch['rgb_img'].numel() > 0 else None

                    outputs = POLICY(depth_img, state_data, rgb_img)
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item()
                    eval_steps += 1

            avg_eval_loss = eval_loss / eval_steps
            print(f"Evaluation at Epoch {epoch}: Avg Loss = {avg_eval_loss:.4f}")
            wandb.log({"periodic_eval_loss": avg_eval_loss, "epoch": epoch})

    # --- Save Results ---
    print("Training finished. Saving model and results...")
    POLICY.release(SDIR)

    save_result(SDIR,
                "loss_policy_origin",
                {"train": train_loss_policy_o, "valid": valid_loss_policy_o}
                )

    # Save network settings and parameters
    dict_layer = POLICY.nn_structure()

    dict_params = {
        'network': args.network,
        'cnn_head': args.cnn_head,
        'use_rgb': args.use_rgb,
        'batch_size': args.batch,
        'sequence_length': args.sequence,
        'hidden_units': args.hidden,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'total_params': POLICY.count_params(),
        'depth_cnn_params': POLICY.depth_cnn_head.count_params(),
        'state_mlp_params': POLICY.state_mlp_head.count_params(),
    }
    if args.use_rgb:
        dict_params['rgb_cnn_params'] = POLICY.rgb_cnn_head.count_params()

    dict_whole = {
        'layer_information': dict_layer,
        'param_information': dict_params
    }

    with open(os.path.join(SDIR, 'network_settings.json'), 'w') as f:
        json.dump(dict_whole, f, indent=4)

    end_time = time.time()
    execution_time = end_time - start_time
    hours = execution_time//3600
    mins = (execution_time % 3600) // 60
    seconds = (execution_time % 3600) % 60

    summary_message = f"Total epochs run: {epoch}. Training finished in " \
           + str(hours) + "h " + str(mins) + "min " + str(seconds) + "s"
    print(summary_message)

    with open(os.path.join(SDIR, "summary.txt"), 'w') as f:
        f.write(summary_message)
    
    wandb.finish()
