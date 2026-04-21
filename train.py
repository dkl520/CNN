import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib

# 在没有 GUI（图形界面）的环境（如远程服务器、Docker 容器）中运行 matplotlib 时，
# 必须使用 'Agg' 后端，否则在尝试绘图时会抛出错误。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
from torch.cuda.amp import autocast, GradScaler  # 用于自动混合精度训练，提升速度并节省内存

# 忽略不影响程序运行的警告信息，保持控制台整洁
warnings.filterwarnings("ignore")

# 假设当前目录下有 model.py，其中定义了 ResNet18 网络结构
from model import ResNet18

# ─────────────────────────────────────────────────────
# 0. 超参数与全局配置 (CFG)
# ─────────────────────────────────────────────────────
CFG = {
    "seed": 42,  # 随机种子，确保实验可重复性
    "batch_size": 128,  # 每个训练批次包含的图像数量
    "epochs": 50,  # 总共训练 50 轮
    "lr": 0.1,  # 初始学习率，对于带动量的 SGD 来说 0.1 是经典起始值
    "momentum": 0.9,  # 动量系数，加速梯度下降并减少振荡
    "weight_decay": 5e-4,  # L2 正则化权重衰减，防止模型过拟合
    "warmup_epochs": 5,  # 前 5 轮进行学习率预热（Warmup）
    "label_smooth": 0.1,  # 标签平滑系数，防止模型过度拟合独热标签（One-hot）
    "num_classes": 10,  # CIFAR-10 数据集有 10 个类别
    "num_workers": 2,  # 加载数据时的并行线程数
    "device": None,  # 运行设备（将在 resolve_device 函数中确定）
    "save_dir": "./checkpoints",  # 权重保存目录
    "history_file": "training_history.json",
    "summary_file": "training_summary.json",
    "report_file": "classification_report.txt",
    "last_ckpt_file": "last_model.pth",
}

# CIFAR-10 对应的类别标签名称
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# 设置随机种子，保证每次运行生成的随机数一致
torch.manual_seed(CFG["seed"])
# 当输入数据的尺寸（Size）固定时，开启此项可以让 cuDNN 自动寻找最高效的卷积算法
torch.backends.cudnn.benchmark = True
os.makedirs(CFG["save_dir"], exist_ok=True)


def resolve_device():
    """
    自动检测当前环境的最优计算设备。
    优先级：NVIDIA GPU (CUDA) > Apple Silicon GPU (MPS) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────
# 1. 数据增强 (Data Augmentation)
# ─────────────────────────────────────────────────────
class Cutout:
    """
    Cutout 增强技术：在图像中随机选择一个矩形区域并将其像素值设为 0。
    这能强迫神经网络去关注图像的其他局部特征，提高泛化能力。
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes  # 遮挡块的数量
        self.length = length  # 遮挡块的边长

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w)
        for _ in range(self.n_holes):
            # 随机选取中心点
            cy = torch.randint(h, (1,)).item()
            cx = torch.randint(w, (1,)).item()
            # 确定遮挡范围，注意边界处理
            y1, y2 = max(0, cy - self.length // 2), min(h, cy + self.length // 2)
            x1, x2 = max(0, cx - self.length // 2), min(w, cx + self.length // 2)
            mask[y1:y2, x1:x2] = 0
        mask = mask.expand_as(img)
        return img * mask


# CIFAR-10 官方统计的像素均值和标准差，用于标准化数据（使输入分布更接近标准正态分布）
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

# 训练集转换：包含多种数据增强手段，防止模型“死记硬背”
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 填充 4 像素后随机裁剪，相当于平移变换
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),  # 随机调整亮度、对比度、饱和度和色调
    transforms.ToTensor(),  # 转换为张量，且像素值缩放到 [0, 1]
    transforms.Normalize(MEAN, STD),  # 标准化处理
    Cutout(n_holes=1, length=16),  # 应用 Cutout 遮挡
])

# 测试集/验证集转换：不进行随机增强，仅需保持数据分布与训练集一致
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ─────────────────────────────────────────────────────
# 2. 损失函数 (Loss Function)
# ─────────────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失函数。
    传统的 CrossEntropy 使用 One-hot 标签（[0, 1, 0...]），这可能导致模型过度自信。
    标签平滑会将部分概率分配给错误类别，使模型输出的概率分布更平滑，提高稳健性。
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_prob = torch.log_softmax(pred, dim=-1)  # 先对输出求 Log Softmax
        # 计算负对数似然（NLL Loss）
        nll = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        # 计算预测分布与均匀分布的距离
        smooth = -log_prob.mean(dim=-1)
        # 按照权重融合两部分损失
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


# 提前定义全局变量，方便在不同函数间调用
model = None
optimizer = None
scheduler = None
criterion = None
scaler = None


# ─────────────────────────────────────────────────────
# 3. 辅助工具函数
# ─────────────────────────────────────────────────────
def warmup_lr(epoch):
    """
    线性 Warmup 策略。
    在训练最初的几个 Epoch，让学习率从一个极小值线性增长到初始学习率。
    这能有效防止训练初期梯度过大导致模型崩坏。
    """
    if epoch < CFG["warmup_epochs"]:
        for g in optimizer.param_groups:
            # 按照比例 (当前轮次/总预热轮次) 调整学习率
            g["lr"] = CFG["lr"] * (epoch + 1) / CFG["warmup_epochs"]


def topk_accuracy(output, target, topk=(1, 5)):
    """
    计算前 K 个预测值中的准确率。
    - Top-1: 预测概率最大的类别是否正确。
    - Top-5: 预测概率前 5 大的类别中是否包含正确类别。
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # 获取前 K 个最大概率的索引
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置以便于与 target 进行比较
        # 判断预测是否等于目标标签
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # 分别计算不同 K 值的准确率
        return [correct[:k].reshape(-1).float().sum(0).item() * 100 / batch_size for k in topk]


def save_json(data, path):
    """保存字典数据到 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_checkpoint_payload(epoch, val_acc):
    """构建保存模型时需要的字典信息包（包含权重和元数据）"""
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "acc": val_acc,
        "optimizer": optimizer.state_dict(),
        "class_names": CIFAR10_CLASSES,
        "mean": list(MEAN),
        "std": list(STD),
        "architecture": "ResNet18",
        "num_classes": CFG["num_classes"],
    }


def run_epoch(loader, train=True):
    """
    运行一个完整的 Epoch（训练或验证）。
    :param loader: 数据加载器
    :param train: True 为训练模式，False 为评估模式
    """
    global model, optimizer, criterion, scaler
    model.train() if train else model.eval()

    total_loss, top1_sum, top5_sum, count = 0.0, 0.0, 0.0, 0

    # 自动混合精度（AMP）目前仅在 CUDA 设备上加速效果明显
    device_type = "cuda" if "cuda" in str(CFG["device"]) else "cpu"

    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG["device"]), labels.to(CFG["device"])

        # 使用 autocast 开启自动混合精度训练（FP32/FP16 自动切换）
        # 这在现代 NVIDIA GPU 上能显著提高吞吐量
        with autocast(enabled=(device_type == "cuda")):
            logits = model(imgs)  # 前向传播
            loss = criterion(logits, labels)  # 计算损失

        if train:
            optimizer.zero_grad(set_to_none=True)  # 梯度清零，set_to_none 更快
            scaler.scale(loss).backward()  # 缩放损失并进行反向传播
            scaler.unscale_(optimizer)  # 在梯度裁剪前反缩放
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
            scaler.step(optimizer)  # 更新权重
            scaler.update()  # 更新缩放器权重

        # 累计统计数据
        t1, t5 = topk_accuracy(logits, labels, topk=(1, 5))
        total_loss += loss.item() * labels.size(0)
        top1_sum += t1 * labels.size(0) / 100
        top5_sum += t5 * labels.size(0) / 100
        count += labels.size(0)

    # 返回平均损失、平均 Top-1 准确率和平均 Top-5 准确率
    return total_loss / count, 100 * top1_sum / count, 100 * top5_sum / count


# ─────────────────────────────────────────────────────
# 4. 主程序流程
# ─────────────────────────────────────────────────────
def main():
    global model, optimizer, scheduler, criterion, scaler

    # 1. 设备检查
    CFG["device"] = resolve_device()
    use_cuda = CFG["device"] == "cuda"
    print(f"▶ 设备: {CFG['device']} (cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()})")

    if CFG["device"] == "cpu":
        raise RuntimeError("未检测到可用 GPU（CUDA 或 MPS）。建议在 GPU 环境下运行。")

    # 2. 数据准备
    # 加载训练集并进行随机分割，分出 10% 作为验证集
    full_train = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)

    val_size = int(0.1 * len(full_train))  # 5000 张
    train_size = len(full_train) - val_size  # 45000 张
    train_set, val_set = random_split(full_train, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(CFG["seed"]))

    # 创建 DataLoader，pin_memory=True 能加速数据从 CPU 到 GPU 的传输
    train_loader = DataLoader(train_set, CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=use_cuda)
    val_loader = DataLoader(val_set, CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=use_cuda)
    test_loader = DataLoader(test_set, CFG["batch_size"], shuffle=False,
                             num_workers=CFG["num_workers"], pin_memory=use_cuda)

    print(f"▶ 训练: {train_size} | 验证: {val_size} | 测试: {len(test_set)}")

    # 3. 初始化模型、优化器和调度器
    model = ResNet18(num_classes=CFG["num_classes"]).to(CFG["device"])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"▶ ResNet-18 可训练参数量: {total_params / 1e6:.2f}M\n")

    criterion = LabelSmoothingCrossEntropy(CFG["label_smooth"])
    # Nesterov 动量 SGD 是训练图像分类模型的强力选择
    optimizer = optim.SGD(model.parameters(), lr=CFG["lr"],
                          momentum=CFG["momentum"], weight_decay=CFG["weight_decay"],
                          nesterov=True)
    # 使用余弦退火算法动态调整学习率，从初始 LR 降至 1e-5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"] - CFG["warmup_epochs"], eta_min=1e-5)
    scaler = GradScaler(enabled=use_cuda)  # 混合精度梯度缩放器

    # 初始化历史记录字典
    history = {k: [] for k in ["tr_loss", "tr_top1", "tr_top5", "va_loss", "va_top1", "va_top5", "lr"]}
    best_acc, best_epoch = 0, 0

    # 打印表格头部
    print("=" * 75)
    print(
        f"{'Epoch':>5} | {'LR':>8} | {'TrLoss':>7} | {'Tr@1':>6} | {'Tr@5':>6} | {'VaLoss':>7} | {'Va@1':>6} | {'Va@5':>6} | {'Best':>5}")
    print("=" * 75)

    # 4. 训练核心循环
    for epoch in range(CFG["epochs"]):
        t0 = time.time()
        warmup_lr(epoch)  # 每一轮开始前检查是否需要 Warmup
        current_lr = optimizer.param_groups[0]["lr"]

        # 执行训练步
        tr_loss, tr_top1, tr_top5 = run_epoch(train_loader, train=True)
        # 执行验证步
        va_loss, va_top1, va_top5 = run_epoch(val_loader, train=False)

        # 过了预热期后，开始执行余弦退火调度
        if epoch >= CFG["warmup_epochs"]:
            scheduler.step()

        # 记录本轮数据
        epoch_results = [tr_loss, tr_top1, tr_top5, va_loss, va_top1, va_top5, current_lr]
        for k, v in zip(history.keys(), epoch_results):
            history[k].append(v)

        # 如果当前验证集准确率是历史最高，保存模型权重
        flag = ""
        if va_top1 > best_acc:
            best_acc, best_epoch = va_top1, epoch + 1
            torch.save(build_checkpoint_payload(epoch + 1, va_top1),
                       f"{CFG['save_dir']}/best_model.pth")
            flag = " ★"  # 在输出行打个星号标记

        elapsed = time.time() - t0
        # 格式化打印当前 Epoch 的状态
        print(
            f"{epoch + 1:>5} | {current_lr:>8.5f} | {tr_loss:>7.4f} | {tr_top1:>5.2f}% | {tr_top5:>5.2f}% | {va_loss:>7.4f} | {va_top1:>5.2f}% | {va_top5:>5.2f}% | {best_acc:>4.1f}%{flag}  [{elapsed:.1f}s]")

    print("=" * 75)
    print(f"\n✅ 训练完成！最优验证准确率: {best_acc:.2f}% (Epoch {best_epoch})\n")

    # 5. 最终性能评估
    # 加载表现最好的那一轮权重，进行最终的测试集测试
    ckpt = torch.load(f"{CFG['save_dir']}/best_model.pth", map_location=CFG["device"])
    model.load_state_dict(ckpt["model"])
    te_loss, te_top1, te_top5 = run_epoch(test_loader, train=False)
    print(f"📊 最终测试集表现 → Loss: {te_loss:.4f} | Top-1: {te_top1:.2f}% | Top-5: {te_top5:.2f}%")

    # 6. 生成混淆矩阵与分类报告
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs.to(CFG["device"]))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n📋 详细分类报告：")
    report = classification_report(all_labels, all_preds, target_names=CIFAR10_CLASSES)
    print(report)

    # 绘制混淆矩阵图
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix (Test Top-1: {te_top1:.2f}%)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    # 7. 绘制训练曲线
    epochs_x = range(1, CFG["epochs"] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # 损失曲线
    axes[0].plot(epochs_x, history["tr_loss"], label="Train")
    axes[0].plot(epochs_x, history["va_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend();
    axes[0].grid(True)
    # 准确率曲线
    axes[1].plot(epochs_x, history["tr_top1"], label="Train Top-1")
    axes[1].plot(epochs_x, history["va_top1"], label="Val Top-1")
    axes[1].set_title("Top-1 Accuracy (%)")
    axes[1].legend();
    axes[1].grid(True)
    # 学习率变化曲线
    axes[2].plot(epochs_x, history["lr"])
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")  # 对数坐标更清晰地观察 LR 衰减
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("▶ 训练曲线与混淆矩阵已保存至本地。")

    # 8. 保存元数据和总结
    history_path = os.path.join(CFG["save_dir"], CFG["history_file"])
    summary_path = os.path.join(CFG["save_dir"], CFG["summary_file"])
    report_path = os.path.join(CFG["save_dir"], CFG["report_file"])
    last_ckpt_path = os.path.join(CFG["save_dir"], CFG["last_ckpt_file"])

    save_json(history, history_path)
    save_json({
        "best_epoch": best_epoch,
        "best_val_top1": round(best_acc, 4),
        "test_loss": round(te_loss, 4),
        "test_top1": round(te_top1, 4),
        "test_top5": round(te_top5, 4),
        "class_names": CIFAR10_CLASSES,
        "num_epochs": CFG["epochs"],
        "device": CFG["device"],
        "model_name": "ResNet18",
    }, summary_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 保存最后一轮的模型作为备份
    torch.save(build_checkpoint_payload(CFG["epochs"], te_top1), last_ckpt_path)
    print(f"▶ 所有训练日志与模型已保存到 {CFG['save_dir']} 目录。")


if __name__ == "__main__":
    main()