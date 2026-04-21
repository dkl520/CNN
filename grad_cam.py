"""
grad_cam.py — Grad-CAM 可视化（CNN 在"看哪里"）
=================================================
Grad-CAM (Gradient-weighted Class Activation Mapping)
原理：
  1. 对目标类别的 logit 求反向传播
  2. 获取最后一个卷积层的梯度 (∂y_c / ∂A^k)
  3. 对梯度做全局平均池化得到权重 α_k
  4. 对特征图加权求和，再经 ReLU 得到热力图

  CAM = ReLU( Σ_k  α_k · A^k )

使用方法：
  python grad_cam.py --ckpt ./checkpoints/best_model.pth
"""

import argparse, torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model import ResNet18

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD  = np.array([0.2023, 0.1994, 0.2010])


# ─────────────────────────────────────────────────────
# Grad-CAM Hook 机制
# ─────────────────────────────────────────────────────
class GradCAM:
    """
    通用 Grad-CAM 实现，用 PyTorch hook 抓取中间层梯度和特征图。

    使用方式：
        cam = GradCAM(model, target_layer=model.layer4[-1].conv2)
        heatmap = cam(input_tensor, class_idx)  # class_idx=None 时自动用预测类别
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        # 注册 forward hook：保存特征图
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        # 注册 backward hook：保存梯度
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # 对目标类别的 logit 反向传播
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # α_k = GAP(梯度)  → [num_channels]
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 加权特征图求和 + ReLU
        cam = (alpha * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # 归一化到 [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx, torch.softmax(logits, dim=1)[0, class_idx].item()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ─────────────────────────────────────────────────────
# 反归一化（把 Tensor 还原为可视化的 RGB 图片）
# ─────────────────────────────────────────────────────
def denormalize(tensor):
    """Tensor [C,H,W] → numpy [H,W,C] uint8"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * STD + MEAN).clip(0, 1)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(img_np, cam, alpha=0.45):
    """
    把 Grad-CAM 热力图叠加到原图上。
    cam 是 [H',W'] 的归一化 float，需要 resize 到原图尺寸。
    """
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap     = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay     = (alpha * heatmap_rgb + (1 - alpha) * img_np).astype(np.uint8)
    return overlay, cam_resized


# ─────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────
def main(ckpt_path, num_images=16, device="cpu"):
    # 加载模型
    model = ResNet18().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ 加载模型：Epoch={ckpt['epoch']}, Val Acc={ckpt['acc']:.2f}%")

    # 挂载 Grad-CAM（监听 layer4 最后一个 Block 的第二个 Conv）
    target_layer = model.layer4[-1].conv2
    gradcam      = GradCAM(model, target_layer)

    # 加载测试集（无增强）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010])
    ])
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    # 随机抽取 num_images 张图
    indices = torch.randperm(len(test_set))[:num_images]

    rows = num_images // 4
    fig, axes = plt.subplots(rows, 8, figsize=(20, rows * 5))
    fig.suptitle("Grad-CAM 可视化：左=原图，右=热力图叠加", fontsize=14, y=1.01)

    for i, idx in enumerate(indices):
        img_tensor, true_label = test_set[idx]
        inp = img_tensor.unsqueeze(0).to(device)
        inp.requires_grad_(True)

        cam, pred_idx, confidence = gradcam(inp)

        img_np          = denormalize(img_tensor)
        overlay, cam_rs = overlay_heatmap(img_np, cam)

        row = i // 4
        col = (i % 4) * 2

        axes[row][col].imshow(img_np)
        axes[row][col].set_title(f"真值: {CIFAR10_CLASSES[true_label]}", fontsize=8)
        axes[row][col].axis("off")

        correct = (pred_idx == true_label)
        color   = "green" if correct else "red"
        axes[row][col+1].imshow(overlay)
        axes[row][col+1].set_title(
            f"预测: {CIFAR10_CLASSES[pred_idx]}\n置信度: {confidence:.1%}",
            fontsize=8, color=color)
        axes[row][col+1].axis("off")

    gradcam.remove_hooks()
    plt.tight_layout()
    plt.savefig("gradcam_visualization.png", dpi=150, bbox_inches="tight")
    print("▶ Grad-CAM 已保存 → gradcam_visualization.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default="./checkpoints/best_model.pth")
    parser.add_argument("--n",      type=int, default=16, help="可视化图片数量（4的倍数）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args.ckpt, args.n, args.device)
