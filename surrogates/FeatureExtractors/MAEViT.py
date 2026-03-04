import torch

# from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, Dinov2Model
from transformers import AutoImageProcessor, ViTMAEForPreTraining
# from .Base import BaseFeatureExtractor
from surrogates.FeatureExtractors.Base import BaseFeatureExtractor
from torchvision import transforms
import torch.nn.functional as F

class DINOv2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(DINOv2FeatureExtractor, self).__init__()
        path = 'D:/code/FOA-Attack-main/vit-mae-base'
        self.model = ViTMAEForPreTraining.from_pretrained(path)
        self.processor = AutoImageProcessor.from_pretrained(path)
        # self.tokenizer = AutoTokenizer.from_pretrained(''facebook/vit-mae-base'')
        self.normalizer = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            # transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    def forward(self, x):

        # outputs = self.model.vit(**x)

        pixel_values = self.normalizer(x)
        outputs = self.model.vit(pixel_values=pixel_values)

        last_hidden_states = outputs.last_hidden_state
        print(last_hidden_states.size())
        image_features = last_hidden_states[:, 0, :]  # torch.Size([1, 768])
        print(image_features.size())
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features



if __name__ == "__main__":
    from PIL import Image
    image1 = Image.open('D:/code/FOA-Attack-main/assets/1.jpg').convert("RGB")
    image2 = Image.open('D:/code/FOA-Attack-main/assets/0.png').convert("RGB")
    transform = transforms.Compose([
        # transforms.Resize([512, 512]),
        transforms.ToTensor(),  # 将 PIL Image 转为 [C, H, W] 的 tensor，像素值归一化到 [0,1]
        # 如果需要归一化到 [-1,1]，可以添加下面一行
        # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # 3. 应用转换
    img_tensor = transform(image1).unsqueeze(0).cuda()
    img_tensor2 = transform(image2).unsqueeze(0).cuda()

    extractor = DINOv2FeatureExtractor()
    extractor = extractor.to("cuda")
    feat1 = extractor(img_tensor * 255)
    feat2 = extractor(img_tensor2 * 255)

    # inputs1 = extractor.processor(images=image1, return_tensors="pt").to("cuda")
    # inputs2 = extractor.processor(images=image2, return_tensors="pt").to("cuda")
    # print(inputs1)
    # feat1 = extractor(inputs1)
    # feat2 = extractor(inputs2)

    sim = F.cosine_similarity(feat1, feat2, dim=-1)
    print(sim)






