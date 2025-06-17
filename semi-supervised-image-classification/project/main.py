# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
from model import WideResNet
from config import config

# 创建日志目录
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 数据增强
class WeakStrongAugment:
    def __init__(self):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

    def __call__(self, x):
        return self.weak(x), self.strong(x)

# 构造半监督数据集
class SemiSupervisedDataset(Dataset):
    def __init__(self, base_dataset):
        np.random.seed(0)
        labeled_idx = np.random.choice(len(base_dataset), config.num_labeled, replace=False)
        self.labeled_data = [(base_dataset[i][0], base_dataset[i][1]) for i in labeled_idx]
        self.unlabeled_data = [base_dataset[i][0] for i in range(len(base_dataset)) if i not in labeled_idx]
        self.transform = WeakStrongAugment()

    def __len__(self):
        return max(len(self.labeled_data), len(self.unlabeled_data))

    def __getitem__(self, idx):
        labeled_idx = idx % len(self.labeled_data)
        unlabeled_idx = idx % len(self.unlabeled_data)

        labeled_img, label = self.labeled_data[labeled_idx]
        unlabeled_img = self.unlabeled_data[unlabeled_idx]
        weak_aug, strong_aug = self.transform(unlabeled_img)

        return (transforms.ToTensor()(labeled_img), label), (weak_aug, strong_aug)

# 测试评估函数
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def main():
    # 数据加载
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(config.mean, config.std)
                                            ]))

    semi_train_set = SemiSupervisedDataset(train_set)
    labeled_loader = DataLoader(semi_train_set, batch_size=config.labeled_batch_size,
                                shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    model = WideResNet(num_classes=config.num_classes).to(config.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    patience = 5  #早停容忍5轮
    trigger = 0
    train_losses, test_accs = [], []

    for epoch in range(config.epochs):
        model.train()
        total_loss, sup_loss, unsup_loss = 0, 0, 0

        pbar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for (x_l, y_l), (x_weak, x_strong) in pbar:
            x_l, y_l = x_l.to(config.device), y_l.to(config.device)
            x_weak, x_strong = x_weak.to(config.device), x_strong.to(config.device)

            logits_l = model(x_l)
            loss_sup = F.cross_entropy(logits_l, y_l)

            with torch.no_grad():
                pseudo_logits = model(x_weak)
                pseudo_probs = torch.softmax(pseudo_logits, dim=-1)
                max_probs, pseudo_labels = torch.max(pseudo_probs, dim=-1)
                mask = (max_probs > config.threshold).float()

            logits_u = model(x_strong)
            loss_unsup = (F.cross_entropy(logits_u, pseudo_labels, reduction='none') * mask).mean()

            loss = loss_sup + config.lambda_u * loss_unsup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            sup_loss += loss_sup.item()
            unsup_loss += loss_unsup.item()

        scheduler.step()

        acc = evaluate(model, test_loader)
        test_accs.append(acc)
        train_losses.append(total_loss / len(labeled_loader))

        print(f"Epoch {epoch+1}: Total Loss={total_loss:.4f}, Acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            trigger = 0
            torch.save(model.state_dict(), f"checkpoints/best_fixmatch_{config.num_labeled}.pth")
        else:
            trigger += 1

        if trigger >= patience:
            print("[Early Stop] No improvement after {} epochs".format(patience))
            break

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.savefig(f"results/fixmatch_curve_{config.num_labeled}.png")
    plt.close()

    print(f"Training Finished. Best Test Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
