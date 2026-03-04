import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor, Siglip2Model, Siglip2Processor, Siglip2ImageProcessor

from .Base import BaseFeatureExtractor
from torchvision import transforms


class Siglip2B32FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(Siglip2B32FeatureExtractor, self).__init__()
        ckpt = "google/siglip2-base-patch32-256"
        self.model = AutoModel.from_pretrained(ckpt)
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.normalizer = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # SigLIP2 imgs mean and std.
        ]
    )

    def forward(self, x):
        # x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs) #torch.Size([1, 1024])
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def text_features_get(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(self.model.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

