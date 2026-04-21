# 🔬 高级 CNN 项目：CIFAR-10 分类 + Grad-CAM 可视化

基于自建 **ResNet-18**（残差网络），在 CIFAR-10 上训练图像分类器，目标准确率 ≥ 92%。
包含工业级训练技巧 + Grad-CAM 可解释性分析。

---

## 📁 项目结构

```
cnn_project_advanced/
├── model.py          # ResNet-18 / ResidualBlock / BottleneckBlock
├── train.py          # 完整训练流程（含 AMP / LR Schedule / 混淆矩阵）
├── grad_cam.py       # Grad-CAM 可视化（CNN 在"看哪里"）
├── app.py            # Flask Web 页面，支持上传图片并预测
├── templates/
│   └── index.html    # 上传图片 + 概率直方图页面
├── requirements.txt  # 依赖
└── checkpoints/      # 自动创建，存放最优模型权重
```

---

## 🚀 快速开始

```bash
pip install -r requirements.txt   # 安装依赖
python train.py                   # 开始训练（CPU ~2h，GPU ~10min）
python grad_cam.py                # 生成 Grad-CAM 热力图
python app.py                     # 启动前端页面
```

---

## 🏗️ 网络架构

```
输入 [B × 3 × 32 × 32]  ← CIFAR-10 彩色图
        │
   Conv3×3 → BN → ReLU  ← Stem（针对小尺寸图优化，不用 7×7）
        │
   ┌────┴────────────────────────┐
   │  ResidualBlock × 2          │  Layer1: 64通道
   │  ┌──────────────┐           │
   │  │  Conv → BN   │           │
   │  │  → ReLU      │           │
   │  │  Conv → BN   │           │
   │  └──────┬───────┘           │
   │      [+] ←── Shortcut       │  ← 核心：x + F(x)
   └─────────────────────────────┘
        │
   Layer2: 128通道, stride=2（下采样）
   Layer3: 256通道, stride=2
   Layer4: 512通道, stride=2
        │
   AdaptiveAvgPool → Flatten → FC(512→10)
        │
   输出 10类 logits
```

---

## ⚙️ 高级训练技巧一览

| 技巧 | 原理 | 效果 |
|------|------|------|
| **残差连接** | x + F(x)，梯度直通高速路 | 可训练更深网络，缓解梯度消失 |
| **BatchNorm** | 每层归一化激活分布 | 加速收敛，可用更大 LR |
| **数据增强** | RandomCrop + Flip + ColorJitter | 扩充训练集，防过拟合 |
| **Cutout** | 随机遮挡一块区域 | 防止网络依赖局部特征 |
| **Label Smoothing** | 软化 one-hot 标签 | 防止过拟合，提升泛化 |
| **SGD + Nesterov** | 带预见性的动量优化 | CV任务上通常比Adam最终精度高 |
| **Cosine Annealing** | LR 按余弦曲线从高降到低 | 避免陷入局部最优 |
| **Linear Warmup** | 前5轮 LR 从0线性增长 | 防止训练初期梯度爆炸 |
| **混合精度 AMP** | FP16 计算 + FP32 参数 | GPU 训练速度提升 2-3× |
| **梯度裁剪** | ‖g‖>1 时等比缩放梯度 | 防止梯度爆炸 |

---

## 🔍 Grad-CAM 解读

运行 `grad_cam.py` 后，你会看到每张图"热力叠加版"：

- **红色区域** = 网络关注最多的区域（对该类别预测贡献最大）
- **蓝色区域** = 网络几乎不关注的背景
- **绿色标题** = 预测正确 / **红色标题** = 预测错误

**示例解读**：
- 预测 `cat` → 热力图集中在耳朵、眼睛 ✅ 合理
- 预测 `automobile` → 热力图集中在车轮、车灯 ✅ 合理
- 预测错误 → 热力图可能集中在背景 ← 说明模型被干扰

---

## 📈 预期结果

| 条件 | Top-1 准确率 |
|------|-------------|
| epoch=50, CPU | ~88-90% |
| epoch=50, GPU | ~92-93% |

---

## 💾 训练经验保存

训练结束后会把模型和训练过程一起保存到 `checkpoints/`：

- `best_model.pth`：最佳验证集模型，可直接用于推理页面
- `last_model.pth`：最后一轮模型
- `training_history.json`：每一轮的 loss / accuracy / lr
- `training_summary.json`：最佳轮次、测试集指标、类别信息
- `classification_report.txt`：测试集分类报告

---

## 🌐 Web 图片预测页面

1. 先训练出模型：

```bash
python train.py
```

2. 启动页面：

```bash
python app.py
```

3. 打开浏览器访问：

```text
http://127.0.0.1:5000
```

页面支持：

- 上传本地图片
- 后端 Python 调用训练好的模型预测类别概率
- 展示“类别 A/B/... 各占多少概率”
- 用直方图显示所有类别概率分布

---

## 🧪 进一步探索

```python
# 1. 换更深的网络
model = ResNet34()   # 在 model.py 中已实现

# 2. 更强的数据增强（需安装 torchvision >= 0.12）
transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)

# 3. 可视化不同层的 Grad-CAM（对比浅层/深层关注点差异）
target_layer = model.layer2[-1].conv2   # 更浅的层
```
