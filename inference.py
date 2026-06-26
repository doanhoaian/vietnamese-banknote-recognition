from dataclasses import dataclass

import torch
from PIL import Image

import config
from data import infer_transform
from model import build_model
from utils import get_device


@dataclass
class Prediction:
    cls: str                  
    label: str                
    confidence: float
    is_money: bool
    is_confident: bool
    topk: list                


def load_checkpoint(device: torch.device, model_path=None):
    path = model_path or config.MODEL_PATH
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(num_classes=len(ckpt["class_dirs"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["class_dirs"]


class CurrencyClassifier:
    def __init__(self, device: torch.device = None, model_path=None):
        self.device = device or get_device()
        self.model, self.class_dirs = load_checkpoint(self.device, model_path)

    @torch.no_grad()
    def predict_probs(self, image: Image.Image) -> torch.Tensor:
        x = infer_transform(image).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0]

    def resolve(self, probs: torch.Tensor, topk: int = 3) -> Prediction:
        conf, idx = probs.max(0)
        return self._resolve(self.class_dirs[idx.item()], conf.item(), probs, topk)

    def predict(self, image: Image.Image, topk: int = 3) -> Prediction:
        return self.resolve(self.predict_probs(image), topk)

    def _resolve(self, cls, conf, probs, topk) -> Prediction:
        is_money = cls != config.NO_MONEY_CLASS
        is_confident = conf >= config.CONF_THRESHOLD

        if not is_money:
            label = config.LABELS_VI[config.NO_MONEY_CLASS]
        elif not is_confident:
            label = config.UNCERTAIN_LABEL
        else:
            label = config.LABELS_VI.get(cls, cls)

        k = min(topk, len(self.class_dirs))
        top = torch.topk(probs, k=k)
        items = [
            (config.LABELS_VI.get(self.class_dirs[i.item()], self.class_dirs[i.item()]),
             p.item())
            for p, i in zip(top.values, top.indices)
        ]
        return Prediction(cls, label, conf, is_money, is_confident, items)
