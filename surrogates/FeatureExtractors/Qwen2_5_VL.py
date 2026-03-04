import sys

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor
from transformers.image_utils import load_images
from qwen_vl_utils import process_vision_info

from torchvision import transforms
# from .Base import BaseFeatureExtractor
from surrogates.FeatureExtractors.Base import BaseFeatureExtractor

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import torch.nn.functional as F


class Qwen2_5_VL_FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(Qwen2_5_VL_FeatureExtractor, self).__init__()
        self.normalizer = transforms.Compose(
        [
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            # transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto")
        # self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")



    def get_input_pixel_values(self, x):
        re_x = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(x)
        pixel_values = self.normalizer(re_x)
        return pixel_values

    # def prepare_inputs(self, x):

    def generate_description(self, image):
        if image.max() == 1.0:
            image = image * 255

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in one concise sentence, no longer than 20 words."},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        print(inputs.pixel_values.size())
        generate_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
        ]
        decoded_output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(decoded_output)

        return decoded_output


    def forward(self, x):
        pixel_values = self.get_input_pixel_values(x)
        pixel_values = pixel_values.to(self.model.device, dtype=torch.bfloat16)
        # t, h, w = pixel_values.size(0), int(pixel_values.size(2) / 14), int(pixel_values.size(3) / 14)
        # image_grid_thw = torch.tensor([[t, h, w]])
        image_grid_thw = torch.tensor([[1, 16, 16]])
        inputs = dict(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        outputs = self.model.get_image_features(**inputs)
        # print(outputs.size())

        image_features = outputs / outputs.norm(dim=1, keepdim=True)
        # print(image_features.size())
        return image_features

    def get_text_embeddings(self, text):

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        # print(inputs)
        inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)  # torch.Size([1, 940, 896])
        # print(inputs_embeds)
        print(inputs_embeds.size())

        return inputs_embeds




    def ceshi_code(self, img_tensor):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the main object in ten words."},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        print(prompt)
        inputs = self.processor(text=prompt, images=[img_tensor*255], return_tensors="pt")
        # inputs = self.processor(text=prompt, return_tensors="pt")
        print(inputs)
        print(inputs.input_ids)
        print(inputs.attention_mask)
        # print(inputs.pixel_values.size())
        x = self.get_input_pixel_values(img_tensor*255)
        print(x.size())
        inputs['pixel_values'] = x
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        generate_ids = self.model.generate(**inputs, max_new_tokens=50)
        decoded_output = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(decoded_output)





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
    img_tensor = transform(image1).unsqueeze(0)
    img_tensor2 = transform(image2).unsqueeze(0)

    extractor = Qwen2_5_VL_FeatureExtractor().eval().to("cuda").requires_grad_(False)

    text = extractor.generate_description(img_tensor*255)
    print(text)
    text_embedding = extractor.get_text_embeddings(text)
    image_features = extractor(img_tensor*255)
    image_features = torch.mean(image_features, dim=0, keepdim=True)
    print(text_embedding.size())
    print(image_features.size())

    # 全局池化
    # img_pool = image_features.mean(dim=1)  # [B, D]
    # ques_pool = text_embedding.mean(dim=1)  # [B, D]

    # 归一化
    img_pool = F.normalize(image_features, dim=-1)
    ques_pool = F.normalize(text_embedding, dim=-1)

    sim_matrix = torch.bmm(img_pool, ques_pool.transpose(1, 2))
    sim_global = sim_matrix.max(dim=1).values.mean(dim=1)
    print(sim_global)

    # # 余弦相似度
    # # sim = (img_pool * ques_pool).sum(dim=-1)  # [B]
    # sim = F.cosine_similarity(img_pool, ques_pool, dim=-1)
    # print(sim)  # 1个样本的相似度




