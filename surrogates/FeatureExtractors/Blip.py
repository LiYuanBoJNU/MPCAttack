import sys

import torch
from transformers import (
    AutoTokenizer,
    Blip2VisionModel,
    Blip2VisionConfig,
    Blip2Processor,
    Blip2Model,
    BlipImageProcessor,
    Blip2ForConditionalGeneration,
)
from torchvision import transforms
# from .Base import BaseFeatureExtractor
from surrogates.FeatureExtractors.Base import BaseFeatureExtractor
device = "cuda" if torch.cuda.is_available() else "cpu"
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class BlipFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(BlipFeatureExtractor, self).__init__()
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)
        # self.generate_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

    def forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(device)
        outputs = self.model.get_image_features(**inputs)
        pooler_output = outputs.pooler_output
        image_features = pooler_output / pooler_output.norm(dim=1, keepdim=True)
        return image_features

    def text_features_get(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(self.model.device)
        text_features = self.model.get_text_features(**inputs)
        print(text_features)
        print(text_features.logits.size())
        text_hidden = text_features.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        print(text_hidden.size())
        sys.exit()
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features


    def generate_description(self, image):

        # prompt = "Please describe the main subject of this image in concise and accurate language."
        prompt = "Question: What is the main subject? Answer:"
        with torch.no_grad():
            # inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.generate_model.device, torch.float16)
            inputs = self.processor(images=image, return_tensors="pt").to(self.generate_model.device, torch.float16)
            generated_ids = self.generate_model.generate(**inputs)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)





if __name__ == "__main__":
    from PIL import Image
    image1 = Image.open('D:/code/FOA-Attack-main/assets/0.png').convert("RGB")
    transform = transforms.Compose([
        # transforms.Resize([512, 512]),
        transforms.ToTensor(),  # 将 PIL Image 转为 [C, H, W] 的 tensor，像素值归一化到 [0,1]
        # 如果需要归一化到 [-1,1]，可以添加下面一行
        # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # 3. 应用转换
    img_tensor = transform(image1).unsqueeze(0).cuda()

    extractor = BlipFeatureExtractor()
    extractor = extractor.to("cuda")
    # extractor.generate_description(img_tensor)
    extractor.text_features_get('a dog')

