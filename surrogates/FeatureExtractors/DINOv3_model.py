import torch

from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, DINOv3ViTModel
# from .Base import BaseFeatureExtractor
from surrogates.FeatureExtractors.Base import BaseFeatureExtractor
from torchvision import transforms
import torch.nn.functional as F

class DINOv3FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(DINOv3FeatureExtractor, self).__init__()
        model_path = 'D:/code/FOA-Attack-main/dinov3-vitb16-pretrain-lvd1689m'
        # model_path = 'D:/code/FOA-Attack-main/dinov3-vitl16-pretrain-lvd1689m'
        self.model = AutoModel.from_pretrained(model_path)
        self.processor = AutoImageProcessor.from_pretrained(model_path, use_safetensors=True)
        # self.tokenizer = AutoTokenizer.from_pretrained('facebook/dinov2-base')
        self.normalizer = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            # transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    def forward(self, x):
        # x = torch.clamp(x, min=0, max=1)
        pixel_values = self.normalizer(x)
        outputs = self.model(pixel_values=pixel_values)
        image_features = outputs.pooler_output
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features


    def ceshi(self,image):
        patch_size = self.model.config.patch_size
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        print(inputs.pixel_values.shape)  # [1, 3, 224, 224]
        batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
        num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
        num_patches_flat = num_patches_height * num_patches_width

        pixel_values=self.normalizer(image).to(self.model.device)
        outputs = self.model(pixel_values=pixel_values)
        print(outputs)
        last_hidden_states = outputs.last_hidden_state
        pooler_output = outputs.pooler_output #pooler_output is the last_hidden_states[:, 0, :]
        print(last_hidden_states.shape)
        print(pooler_output.shape)

        cls_token = last_hidden_states[:, 0, :]
        patch_features = last_hidden_states[:, 1:, :]
        print(cls_token.shape)
        print(patch_features.shape)
        return pooler_output


if __name__ == "__main__":
    from PIL import Image
    image1 = Image.open('D:/code/FOA-Attack-main/assets/0.jpg').convert("RGB")
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

    extractor = DINOv3FeatureExtractor()
    extractor = extractor.to("cuda")
    # feat1 = extractor.ceshi(img_tensor*255)
    # feat2 = extractor.ceshi(img_tensor2*255)
    feat1 = extractor(img_tensor * 255)
    feat2 = extractor(img_tensor2 * 255)
    sim = F.cosine_similarity(feat1, feat2, dim=-1)
    print(sim)
    # sim.backward()
    # g = img_tensor.grad






