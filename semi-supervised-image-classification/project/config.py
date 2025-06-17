# config.py
import torch
import os

class Config:
    # 数据集设置
    dataset = "CIFAR-10"
    num_labeled = 4000      # 标注样本总数，可设置为 40, 250, 4000
    num_classes = 10         # CIFAR-10 类别数

    # 数据归一化参数
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # 训练参数
    epochs = 30
    batch_size = 64
    labeled_batch_size = 16  # 在FixMatch中通常用于Labeled损失
    lr = 0.03
    momentum = 0.9
    weight_decay = 5e-4

    # SSL超参数
    threshold = 0.98         # Pseudo-label的置信度阈值
    lambda_u = 1.0           # 无监督损失权重

    # 提前停止
    early_stop_patience = 5  # 若acc超过5个epoch无提升则提前停止

    # 训练设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4

    # 日志与模型路径
    log_dir = "./logs"
    checkpoint_path = "./checkpoints/best_model.pth"

    # 确保目录存在
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

config = Config()
