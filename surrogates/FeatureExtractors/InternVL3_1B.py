import sys

import torch
from transformers import AutoProcessor, AutoModel, AutoModelForImageTextToText, AutoTokenizer, InternVLModel, InternVLProcessor
from transformers.image_utils import load_images

from torchvision import transforms
# from .Base import BaseFeatureExtractor
from surrogates.FeatureExtractors.Base import BaseFeatureExtractor

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import torch.nn.functional as F

def dynamic_preprocess_tensor(images, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    images: torch.Tensor of shape [B, C, H, W]
    return: list of torch.Tensor, each shape [num_tiles, C, image_size, image_size]
    """
    B, C, H, W = images.shape
    output_list = []

    # for b in range(B):
    img = images[0]  # [C,H,W]
    aspect_ratio = W / H


    target_ratios = sorted(
        [(i, j) for n in range(min_num, max_num+1) for i in range(1,n+1) for j in range(1,n+1)
         if min_num <= i*j <= max_num], key=lambda x: x[0]*x[1])


    target_aspect_ratio = min(target_ratios, key=lambda r: abs((r[0]/r[1]) - aspect_ratio))
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]


    img_resized = F.interpolate(img.unsqueeze(0), size=(int(target_height), int(target_width)), mode='bilinear', align_corners=False)[0]


    tiles = []
    n_w = int(target_width // image_size)
    n_h = int(target_height // image_size)
    for i in range(n_h):
        for j in range(n_w):
            tile = img_resized[:, i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size]
            tiles.append(tile)
    tiles = torch.stack(tiles, dim=0)  # [num_tiles, C, image_size, image_size]


    if use_thumbnail and tiles.shape[0] != 1:
        thumb = F.interpolate(img.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False)[0]
        tiles = torch.cat([tiles, thumb.unsqueeze(0)], dim=0)

    # output_list.append(tiles)

    return tiles

class InternVL3_1B_FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(InternVL3_1B_FeatureExtractor, self).__init__()
        self.normalizer = transforms.Compose(
        [
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            # transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )
        # path = "OpenGVLab/InternVL3-1B"
        # self.processor = AutoProcessor.from_pretrained(path)
        # self.tokenizer = AutoTokenizer.from_pretrained(path)
        # self.model = AutoModel.from_pretrained(
        #     path,
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     use_flash_attn=True,
        #     trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
        self.model = AutoModelForImageTextToText.from_pretrained("OpenGVLab/InternVL3-1B-hf", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-hf")



    def get_input_pixel_values(self, x):
        re_x = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(x)
        pixel_values = self.normalizer(dynamic_preprocess_tensor(re_x, use_thumbnail=True))
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
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")

        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        generate_ids = self.model.generate(**inputs, max_new_tokens=50)
        decoded_output = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # print('generated description:', decoded_output)
        return decoded_output


    def forward(self, x):
        pixel_values = self.get_input_pixel_values(x)
        pixel_values = pixel_values.to(self.model.device, dtype=torch.bfloat16)
        inputs = dict(pixel_values=pixel_values)
        outputs = self.model.get_image_features(**inputs)
        # print(outputs.size())

        image_features = outputs / outputs.norm(dim=1, keepdim=True)
        # print(image_features.size())
        return image_features

    def global_local_features(self, x):
        pixel_values = self.get_input_pixel_values(x)
        pixel_values = pixel_values.to(self.model.device, dtype=torch.bfloat16)
        # inputs = dict(pixel_values=pixel_values)
        vision_features = self.model.vision_tower(pixel_values=pixel_values).last_hidden_state
        # print(vision_features.size())
        # vision_features = vision_features[:, 1:, :]
        # print(vision_features.size())
        # sys.exit()
        global_feature = vision_features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = vision_features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=1, keepdim=True)
        # features = self.model.get_image_embedding(inputs["pixel_values"])
        # global_feature = features[:, 0, :]
        # local_feature = features[:, 1:, :]
        return global_feature, local_feature



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

