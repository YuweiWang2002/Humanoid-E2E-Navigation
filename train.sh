#!/bin/bash

# ===================================================================
# 训练脚本 - 人形机器人端到端导航模型
# ===================================================================
#
# 使用前请根据需要修改以下参数

python train_all_models.py \
  --name "Humanoid_GRU_Nvidia_Seq16" \
  --network "GRU" \
  --cnn_head "Nvidia" \
  --epoch 150 \
  --batch 32 \
  --sequence 16 \
  --learning_rate 1e-4 \
  --seed 42
  # --use_rgb \ # 如果需要使用RGB图像，请取消此行注释
  # --resume     # 如果需要从断点续训，请取消此行注释