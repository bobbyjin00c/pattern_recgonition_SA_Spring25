import torch

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = None
        self.counter = 0

    def step(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def save_checkpoint(model, acc, best_acc, path="./checkpoint.pth"):
    if acc > best_acc:
        torch.save(model.state_dict(), path)
