import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from nets.models_all import Humanoid_All_Models
from dataset_all import HumanoidNavDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

def test(args):
    """
    对训练好的模型在测试集上进行评估。
    """
    # --- 1. 加载模型和配置 ---
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 加载最佳模型的检查点
    model_path = os.path.join(args.results_dir, args.name, 'best_checkpoint.pth')
    if not os.path.exists(model_path):
        print(f"错误: 在 {model_path} 未找到模型检查点")
        return

    checkpoint = torch.load(model_path, map_location=device)
    
    # 从保存的配置中重新创建模型架构
    config = checkpoint['config']
    model = Humanoid_All_Models(
        cnn_head=config['cnn_head'],
        rnn_type=config['network'],
        use_rgb=config['use_rgb'],
        rnn_hidden_size=config['hidden']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 必须设置为评估模式
    print(f"模型已从 {model_path} 加载")

    # --- 2. 准备测试数据集 ---
    # 重要提示: 测试集不应在训练或验证中使用过！
    # 这里我们使用与训练脚本中验证集相同的分割逻辑来定义测试集。
    # 一个更严谨的方法是预先定义好固定的训练/验证/测试文件列表。
    all_files = sorted([os.path.join(config['processed_dir'], f) for f in os.listdir(config['processed_dir']) if f.endswith('.csv')])
    
    # 根据训练时的验证集比例划分出测试集
    split_idx = int(len(all_files) * (1 - config['valid_split']))
    test_files = all_files[split_idx:]
    
    print(f"找到 {len(test_files)} 个轨迹文件用于测试。")

    test_dataset = HumanoidNavDataset(
        trajectory_files=test_files,
        sequence_length=config['sequence'],
        use_rgb=config['use_rgb']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch'],
        shuffle=False, # 测试时无需打乱
        num_workers=0  # 在Windows上建议为0
    )

    # --- 3. 运行评估循环 ---
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad(): # 关闭梯度计算，节省显存并加速
        for batch in tqdm(test_loader, desc="在测试集上评估"):
            depth_seq = batch['depth_image'].to(device)
            state_seq = batch['state'].to(device)
            gt_actions = batch['action'].to(device)
            rgb_seq = batch.get('rgb_image')
            if rgb_seq is not None:
                rgb_seq = rgb_seq.to(device)

            predicted_actions = model(depth_seq, rgb_seq, state_seq)

            all_predictions.append(predicted_actions.cpu().numpy())
            all_ground_truth.append(gt_actions.cpu().numpy())

    # --- 4. 计算并报告性能指标 ---
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    mse = mean_squared_error(all_ground_truth, all_predictions, multioutput='raw_values')

    print("\n--- 测试集评估结果 ---")
    print(f"实验名称: {args.name}")
    print("-----------------------------------")
    print(f"均方误差 (vel_x):  {mse[0]:.6f}")
    print(f"均方误差 (vel_y):  {mse[1]:.6f}")
    print(f"均方误差 (vel_yaw): {mse[2]:.6f}")
    print("-----------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估已训练的人形机器人导航模型")
    parser.add_argument('--results_dir', type=str, default='results', help='存放训练结果的目录')
    parser.add_argument('--name', type=str, required=True, help='需要评估的训练运行的名称')
    args = parser.parse_args()
    test(args)