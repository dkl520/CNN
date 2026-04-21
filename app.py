import io
import os

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

from model import ResNet18

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
DEFAULT_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
DEFAULT_STD = [0.2023, 0.1994, 0.2010]

app = Flask(__name__)
_predictor = None


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ImagePredictor:
    def __init__(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到模型文件: {checkpoint_path}")

        self.device = resolve_device()
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = self.checkpoint.get("class_names", DEFAULT_CLASSES)
        mean = self.checkpoint.get("mean", DEFAULT_MEAN)
        std = self.checkpoint.get("std", DEFAULT_STD)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.model = ResNet18(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(self.checkpoint["model"])
        self.model.eval()

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

        results = [
            {
                "label": label,
                "probability": round(prob * 100, 4),
            }
            for label, prob in zip(self.class_names, probs)
        ]
        results.sort(key=lambda item: item["probability"], reverse=True)
        return {
            "top_prediction": results[0],
            "predictions": results,
            "model_info": {
                "architecture": self.checkpoint.get("architecture", "ResNet18"),
                "epoch": self.checkpoint.get("epoch"),
                "val_acc": self.checkpoint.get("acc"),
            },
        }


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ImagePredictor(CHECKPOINT_PATH)
    return _predictor


@app.get("/")
def index():
    model_ready = os.path.exists(CHECKPOINT_PATH)
    return render_template("index.html", model_ready=model_ready)


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "请先上传图片文件。"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "上传文件名为空。"}), 400

    try:
        result = get_predictor().predict(file.read())
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"预测失败: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
