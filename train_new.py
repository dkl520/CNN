"""
train.py — M5 Mac MPS 性能修复版
===================================
修复了两个主要性能陷阱：
  Bug 1: torch.autocast(device_type="mps") — MPS float16 算子支持不完整，
          大量算子会 fallback 回 CPU，导致 MPS ↔ CPU 频繁数据搬运，比不开还慢。
  Bug 2: num_workers=8 — macOS 用 spawn 而非 fork 启动子进程，
          8 个子进程的启动和通信开销远超收益，在 M5 上反而拖慢数据加载。

改动 3 处，其他代码保持不变：
  1. num_workers: 8 → 0       （主进程直接加载，零 IPC 开销）
  2. 去掉 MPS autocast         （纯 float32 反而更快）
  3. pin_memory 仅 CUDA 开启   （MPS 不受益）
"""

import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
from model import ResNet18

CFG = {
    "seed": 42,
    "batch_size": 256,
    "epochs": 50,
    "lr": 0.15,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "warmup_epochs": 5,
    "label_smooth": 0.1,
    "num_classes": 10,
    # ✅ Fix 1: 0 → 主进程直接读数据，彻底消灭 macOS spawn 开销
    "num_workers": 0,
    "device": None,
    "save_dir": "./checkpoints",
    "history_file": "training_history.json",
    "summary_file": "training_summary.json",
    "report_file": "classification_report.txt",
    "last_ckpt_file": "last_model.pth",
}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

torch.manual_seed(CFG["seed"])
torch.backends.cudnn.benchmark = True
os.makedirs(CFG["save_dir"], exist_ok=True)


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w)
        for _ in range(self.n_holes):
            cy = torch.randint(h, (1,)).item()
            cx = torch.randint(w, (1,)).item()
            y1, y2 = max(0, cy - self.length // 2), min(h, cy + self.length // 2)
            x1, x2 = max(0, cx - self.length // 2), min(w, cx + self.length // 2)
            mask[y1:y2, x1:x2] = 0
        return img * mask.expand_as(img)


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    Cutout(n_holes=1, length=16),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = torch.log_softmax(pred, dim=-1)
        nll = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_prob.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


model = optimizer = scheduler = criterion = scaler = None


def warmup_lr(epoch):
    if epoch < CFG["warmup_epochs"]:
        for g in optimizer.param_groups:
            g["lr"] = CFG["lr"] * (epoch + 1) / CFG["warmup_epochs"]


def topk_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum(0).item() * 100 / target.size(0) for k in topk]


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_checkpoint_payload(epoch, val_acc):
    return {"epoch": epoch, "model": model.state_dict(), "acc": val_acc,
            "optimizer": optimizer.state_dict(), "class_names": CIFAR10_CLASSES,
            "mean": list(MEAN), "std": list(STD),
            "architecture": "ResNet18", "num_classes": CFG["num_classes"]}


def run_epoch(loader, train=True):
    global model, optimizer, criterion, scaler
    model.train() if train else model.eval()
    is_cuda = "cuda" in str(CFG["device"])
    total_loss, top1_sum, top5_sum, count = 0.0, 0.0, 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG["device"]), labels.to(CFG["device"])

        # ✅ Fix 2: MPS 不开 autocast，直接 float32
        # MPS autocast 会触发大量 CPU fallback，比不开更慢。
        # CUDA 保留 AMP，MPS/CPU 走纯 float32。
        if is_cuda:
            with torch.autocast(device_type="cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if is_cuda:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        t1, t5 = topk_accuracy(logits, labels, topk=(1, 5))
        total_loss += loss.item() * labels.size(0)
        top1_sum += t1 * labels.size(0) / 100
        top5_sum += t5 * labels.size(0) / 100
        count += labels.size(0)

    return total_loss / count, 100 * top1_sum / count, 100 * top5_sum / count


def main():
    global model, optimizer, scheduler, criterion, scaler

    CFG["device"] = resolve_device()
    is_cuda = "cuda" in str(CFG["device"])
    print(f"▶ 设备: {CFG['device']}")

    full_train = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(CFG["seed"]))

    # ✅ Fix 3: pin_memory 仅 CUDA 开启（MPS 不受益）
    pin = is_cuda
    kw = dict(num_workers=CFG["num_workers"], pin_memory=pin)
    train_loader = DataLoader(train_set, CFG["batch_size"], shuffle=True, **kw)
    val_loader = DataLoader(val_set, CFG["batch_size"], shuffle=False, **kw)
    test_loader = DataLoader(test_set, CFG["batch_size"], shuffle=False, **kw)
    print(f"▶ 训练: {train_size} | 验证: {val_size} | 测试: {len(test_set)}")

    model = ResNet18(num_classes=CFG["num_classes"]).to(CFG["device"])
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"▶ ResNet-18 可训练参数: {total / 1e6:.2f}M\n")

    criterion = LabelSmoothingCrossEntropy(CFG["label_smooth"])
    optimizer = optim.SGD(model.parameters(), lr=CFG["lr"],
                          momentum=CFG["momentum"], weight_decay=CFG["weight_decay"],
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"] - CFG["warmup_epochs"], eta_min=1e-5)
    scaler = GradScaler(enabled=is_cuda)

    history = {k: [] for k in ["tr_loss", "tr_top1", "tr_top5", "va_loss", "va_top1", "va_top5", "lr"]}
    best_acc, best_epoch = 0, 0

    print("=" * 75)
    print(f"{'Epoch':>5} | {'LR':>8} | {'TrLoss':>7} | {'Tr@1':>6} | {'Tr@5':>6} "
          f"| {'VaLoss':>7} | {'Va@1':>6} | {'Va@5':>6} | {'Best':>5}")
    print("=" * 75)

    for epoch in range(CFG["epochs"]):
        t0 = time.time()
        warmup_lr(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_top1, tr_top5 = run_epoch(train_loader, train=True)
        va_loss, va_top1, va_top5 = run_epoch(val_loader, train=False)

        if epoch >= CFG["warmup_epochs"]:
            scheduler.step()

        for k, v in zip(history.keys(),
                        [tr_loss, tr_top1, tr_top5, va_loss, va_top1, va_top5, current_lr]):
            history[k].append(v)

        flag = ""
        if va_top1 > best_acc:
            best_acc, best_epoch = va_top1, epoch + 1
            torch.save(build_checkpoint_payload(epoch + 1, va_top1),
                       f"{CFG['save_dir']}/best_model.pth")
            flag = " ★"

        print(f"{epoch + 1:>5} | {current_lr:>8.5f} | {tr_loss:>7.4f} | {tr_top1:>5.2f}% "
              f"| {tr_top5:>5.2f}% | {va_loss:>7.4f} | {va_top1:>5.2f}% | {va_top5:>5.2f}% "
              f"| {best_acc:>4.1f}%{flag}  [{time.time() - t0:.1f}s]")

    print("=" * 75)
    print(f"\n✅ 训练完成！最优验证准确率: {best_acc:.2f}% (Epoch {best_epoch})\n")

    ckpt = torch.load(f"{CFG['save_dir']}/best_model.pth", map_location=CFG["device"])
    model.load_state_dict(ckpt["model"])
    te_loss, te_top1, te_top5 = run_epoch(test_loader, train=False)
    print(f"📊 测试集 → Loss: {te_loss:.4f} | Top-1: {te_top1:.2f}% | Top-5: {te_top5:.2f}%")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs.to(CFG["device"]))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=CIFAR10_CLASSES)
    print("\n📋 分类报告：\n", report)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES, ax=ax)
    ax.set_xlabel("Predicted");
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix (Test Top-1: {te_top1:.2f}%)")
    plt.tight_layout();
    plt.savefig("confusion_matrix.png", dpi=150)

    epochs_x = range(1, CFG["epochs"] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs_x, history["tr_loss"], label="Train")
    axes[0].plot(epochs_x, history["va_loss"], label="Val")
    axes[0].set_title("Loss");
    axes[0].legend();
    axes[0].grid(True)
    axes[1].plot(epochs_x, history["tr_top1"], label="Train Top-1")
    axes[1].plot(epochs_x, history["va_top1"], label="Val Top-1")
    axes[1].set_title("Top-1 Accuracy (%)");
    axes[1].legend();
    axes[1].grid(True)
    axes[2].plot(epochs_x, history["lr"])
    axes[2].set_title("Learning Rate");
    axes[2].set_yscale("log");
    axes[2].grid(True)
    plt.tight_layout();
    plt.savefig("training_curves.png", dpi=150)

    save_json(history, f"{CFG['save_dir']}/{CFG['history_file']}")
    save_json({"best_epoch": best_epoch, "best_val_top1": round(best_acc, 4),
               "test_top1": round(te_top1, 4), "device": str(CFG["device"])},
              f"{CFG['save_dir']}/{CFG['summary_file']}")
    with open(f"{CFG['save_dir']}/{CFG['report_file']}", "w") as f:
        f.write(report)
    torch.save(build_checkpoint_payload(CFG["epochs"], te_top1),
               f"{CFG['save_dir']}/{CFG['last_ckpt_file']}")
    print(f"▶ 全部文件已保存到 {CFG['save_dir']}")


if __name__ == "__main__":
    main()
