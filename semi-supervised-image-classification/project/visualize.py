import re
import matplotlib.pyplot as plt

def parse_logs(log_text):
    epochs = []
    losses = []
    accuracies = []

    # 匹配行如：Epoch 1: Loss=204.0957, Test Acc=19.61%
    pattern = re.compile(r"Epoch (\d+): Loss=([\d.]+), Test Acc=([\d.]+)%")

    for match in pattern.finditer(log_text):
        epoch = int(match.group(1))
        loss = float(match.group(2))
        acc = float(match.group(3))

        epochs.append(epoch)
        losses.append(loss)
        accuracies.append(acc)

    return epochs, losses, accuracies

def plot_logs(log_text, title="Training Curve", save_path="training_curve.png"):
    epochs, losses, accuracies = parse_logs(log_text)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 画 Loss 曲线
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, losses, color="tab:red", label="Loss")
    ax1.tick_params(axis='y', labelcolor="tab:red")

    # 画 Accuracy 曲线，共享 X 轴
    ax2 = ax1.twinx()
    ax2.set_ylabel("Test Accuracy (%)", color="tab:blue")
    ax2.plot(epochs, accuracies, color="tab:blue", label="Accuracy")
    ax2.tick_params(axis='y', labelcolor="tab:blue")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    with open("output/log/usb_log/usb_log250.txt", "r") as f:
        log_text = f.read()
    plot_logs(log_text, title="usb (250 labeled)", save_path="usb_250.png")
