# torchSSL_fm.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms.autoaugment import RandAugment
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import config
from model import WideResNet
from utils import EarlyStopping, save_checkpoint

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
            RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

    def __call__(self, img):
        return self.weak(img), self.strong(img)

class FixMatchDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.augment = WeakStrongAugment()

    def __len__(self):
        return max(len(self.labeled_dataset), len(self.unlabeled_dataset))

    def __getitem__(self, idx):
        labeled_idx = idx % len(self.labeled_dataset)
        unlabeled_idx = idx % len(self.unlabeled_dataset)

        x_l, y_l = self.labeled_dataset[labeled_idx]
        x_u = self.unlabeled_dataset[unlabeled_idx][0]
        x_w, x_s = self.augment(x_u)

        return (x_l, y_l), (x_w, x_s)

def get_dataloaders():
    base_train = datasets.CIFAR10('./data', train=True, download=True)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])
    test_set = datasets.CIFAR10('./data', train=False, transform=test_transform)

    num_classes = config.num_classes
    label_per_class = config.num_labeled // num_classes
    labeled_idxs, unlabeled_idxs = [], []

    class_idxs = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(base_train):
        class_idxs[label].append(idx)

    for c in range(num_classes):
        np.random.shuffle(class_idxs[c])
        labeled_idxs.extend(class_idxs[c][:label_per_class])
        unlabeled_idxs.extend(class_idxs[c][label_per_class:])

    transform = WeakStrongAugment().weak
    labeled_dataset = [(transform(base_train[i][0]), base_train[i][1]) for i in labeled_idxs]
    unlabeled_dataset = [base_train[i] for i in unlabeled_idxs]

    train_dataset = FixMatchDataset(labeled_dataset, unlabeled_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(config.device), y.to(config.device)
            preds = model(x)
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

def plot_accuracy(acc_list):
    plt.figure()
    plt.plot(acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('FixMatch Test Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.log_dir, 'accuracy_curve.png'))
    plt.close()

def main():
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    model = WideResNet(num_classes=config.num_classes).to(config.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_loader, test_loader = get_dataloaders()

    early_stopper = EarlyStopping(patience=config.early_stop_patience)
    best_acc = 0
    acc_list = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for (x_l, y_l), (x_w, x_s) in loop:
            x_l, y_l = x_l.to(config.device), y_l.to(config.device)
            x_w, x_s = x_w.to(config.device), x_s.to(config.device)

            logits_l = model(x_l)
            loss_s = F.cross_entropy(logits_l, y_l)

            with torch.no_grad():
                pseudo_logits = model(x_w)
                pseudo_probs = torch.softmax(pseudo_logits, dim=-1)
                max_probs, pseudo_labels = torch.max(pseudo_probs, dim=-1)
                mask = (max_probs > config.threshold).float()

            logits_u = model(x_s)
            loss_u = (F.cross_entropy(logits_u, pseudo_labels, reduction='none') * mask).mean()
            loss = loss_s + config.lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        acc = evaluate(model, test_loader)
        acc_list.append(acc)

        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Test Acc={acc:.2f}%")
        save_checkpoint(model, acc, best_acc, config.checkpoint_path)
        best_acc = max(best_acc, acc)

        if early_stopper.step(acc):
            print(f"[Early Stop] No improvement for {early_stopper.patience} epochs")
            break

    plot_accuracy(acc_list)
    print(f"Training Finished. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
