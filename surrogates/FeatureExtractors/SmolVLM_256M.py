import sys

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, SmolVLMImageProcessor
# from .Base import BaseFeatureExtractor
from surrogates.FeatureExtractors.Base import BaseFeatureExtractor
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from utils import mmd_loss
import ot
from surrogates.FeatureExtractors.SmolVLM_imgage_process import split_into_17


class SmolVLM_256M_FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(SmolVLM_256M_FeatureExtractor, self).__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            torch_dtype=torch.bfloat16,
            # device_map="auto"
        ).to("cuda")
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        self.normalizer = transforms.Compose(
            [
                # transforms.Resize((512, 512)),
                transforms.Normalize(  # 和模型训练时一致
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ]
        )

        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": 'Describe this image in one concise sentence, no longer than 20 words.'}
                ]
            },
        ]
        Placeholder_image = torch.zeros(1, 3, 512, 512)
        self.prompt = self.processor.apply_chat_template(self.messages, add_generation_prompt=True)
        # self.inputs = self.processor(text=self.prompt, images=[Placeholder_image], return_tensors="pt")
        self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
            self.processor.tokenizer.additional_special_tokens.index("<image>")]


    def prepare_answer(self, target_img):
        tgt_img = target_img / 255
        inputs_img_tgt = self.processor(text=self.prompt, images=[tgt_img], return_tensors="pt")
        inputs_img_tgt = inputs_img_tgt.to(self.model.device, dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs_img_tgt, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        # print(generated_texts)
        text = generated_texts[0]
        answer = text.split("Assistant:")[-1].strip()
        print(answer)
        return answer

    def prepare_labels(self, target_img):
        tgt_img = target_img / 255
        inputs_img_tgt = self.processor(text=self.prompt, images=[tgt_img], return_tensors="pt")
        inputs_img_tgt = inputs_img_tgt.to(self.model.device, dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs_img_tgt, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        # print(generated_texts)
        text = generated_texts[0]
        answer = text.split("Assistant:")[-1].strip()
        print(answer)


        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": 'Describe this image in one concise sentence, no longer than 20 words.'}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        text_with_answer = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        # tokenization
        inputs = self.processor(text=text_with_answer, images=[tgt_img], return_tensors="pt", padding=True)
        # inputs = self.processor(text=text_with_answer, images=[adv_img], return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        # inputs["labels"] = labels
        # self.inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

        return inputs.to(self.model.device, dtype=torch.bfloat16), labels.to(self.model.device)



    def compute_loss(self, logits, labels):
        # 如果 labels 比 logits 长，只计算 logits 对应部分
        # if labels.size(1) > logits.size(1):
        #     active_labels = labels[:, :logits.size(1)]
        # else:
        #     active_labels = labels
        active_labels = labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = active_labels[:, 1:].contiguous()
        # print("logits:", shift_logits.shape)
        # print("labels:", shift_labels.shape)
        # loss = F.cross_entropy(
        #     shift_logits.view(-1, shift_logits.size(-1)),
        #     shift_labels.view(-1),
        #     ignore_index=-100
        # )

        active_loss_mask = shift_labels.view(-1) != -100
        active_logits1 = shift_logits.view(-1, shift_logits.size(-1))[active_loss_mask]
        active_labels1 = shift_labels.view(-1)[active_loss_mask]

        # 交叉熵损失
        loss = F.cross_entropy(active_logits1, active_labels1)

        return loss

    def forward(self, inputs, mode='normal'):
        # Prepare inputs
        # inputs = self.processor(text=self.prompt, images=[x], return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

        # if mode == 'normal':
        #     inputs = self.processor(text=self.prompt, images=[x], return_tensors="pt")
        #     inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        # elif mode == 'adv':
        #     pixel_values = split_into_17(x)
        #     print(pixel_values.size())
        #     self.inputs['pixel_values'] = pixel_values.unsqueeze(0)
        #     inputs = self.inputs.to(self.model.device, dtype=torch.bfloat16)

        outputs = self.model(**inputs)
        logits = outputs.logits
        image_features = outputs.image_hidden_states

        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        output = dict(logits=logits, image_features=image_features)
        return output



    def ceshi(self):
        # from transformers.image_utils import load_image
        # # Load images
        # image1 = load_image(
        #     "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        # image2 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")
        from PIL import Image
        image1 = Image.open('D:/code/FOA-Attack-main/assets/0.png').convert("RGB")
        image2 = Image.open('D:/code/FOA-Attack-main/assets/logo.png').convert("RGB")

        img_path = "D:/code/FOA-Attack-main/assets/0.png"
        img = Image.open(img_path).convert("RGB")  # 转成 RGB，确保通道数为 3

        # 2. 定义转换
        transform = transforms.Compose([
            transforms.Resize([512,512]),
            transforms.ToTensor(),  # 将 PIL Image 转为 [C, H, W] 的 tensor，像素值归一化到 [0,1]
            # 如果需要归一化到 [-1,1]，可以添加下面一行
            # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        # 3. 应用转换
        img_tensor = transform(img).unsqueeze(0)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": 'Describe this image in one concise sentence, no longer than 20 words.'}
                ]
            },
        ]

        # Prepare inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        print(prompt)
        print(img_tensor.size())
        inputs = self.processor(text=prompt, images=[img_tensor], return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        print(inputs)
        # print(img_tensor.unsqueeze(0).size())
        # inputs['pixel_values'] = img_tensor.unsqueeze(0)
        # prompt_len = inputs.input_ids.shape[1]
        # labels = inputs.input_ids.clone()
        # if prompt_len <= labels.shape[1]:
        #     labels[:, :prompt_len] = -100
        # else:
        #     print(
        #         f"Warning: Prompt token length ({prompt_len}) > total label length ({labels.shape[1]}). Check template/answer/truncation.")
        #     labels[:, :] = -100
        # labels[inputs.attention_mask == 0] = -100

        labels = inputs.input_ids.clone()
        print(labels)

        # Generate outputs
        # generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        outputs = self.model(**inputs)
        print(outputs)
        logits = outputs.logits
        print(logits.size())
        print(outputs.image_hidden_states.size())



        sys.exit()
        inputs = self.processor(text=prompt, images=[image2], return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        outputs = self.model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()

        # 假设序列长度不够 max_len，需要 padding
        max_len = shift_logits.size(1)
        padded_labels = torch.full((1, max_len+1), -100).to(self.model.device)
        padded_labels[:, :labels.size(1)] = labels
        labels = padded_labels

        shift_labels = labels[..., 1:].contiguous()
        print(shift_logits.size())
        print(shift_labels.size())

        active_loss_mask = shift_labels.view(-1) != -100
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss_mask]
        active_labels = shift_labels.view(-1)[active_loss_mask]

        if active_labels.numel() == 0:
            print(1)
            loss = torch.tensor(0.0, device=logits.device)
        else:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(active_logits, active_labels)

        print(loss)






        # print(logits2.size())
        # # loss_KL = F.kl_div((logits2.log_softmax(dim=-1)),
        # #                    p_clean, reduction='batchmean')
        # # print(loss_KL)
        # x = torch.randn(1151, 49280, device='cuda')
        # y = torch.randn(887, 49280, device='cuda')
        # # print(mmd_loss(x, y, sigmas=(50, 100, 150, 200, 300)))
        # # loss = mmd_loss(logits.squeeze(), logits2.squeeze())
        # # print(loss)
        #
        # # x = F.normalize(x, dim=1)
        # # y = F.normalize(y, dim=1)
        #
        # x = F.normalize(logits.squeeze(), dim=1)
        # y = F.normalize(logits2.squeeze(), dim=1)
        # # 余弦距离矩阵 [n, m]
        # cost = 1.0 - x @ y.t()  # 值域 [0, 2]
        #
        # # 均匀权重
        # a = torch.ones(x.size(0), device=x.device, dtype=x.dtype) / x.size(0)
        # b = torch.ones(y.size(0), device=y.device, dtype=y.dtype) / y.size(0)
        # # Sinkhorn 迭代得到传输矩阵 T [n, m]
        # T = ot.sinkhorn(a, b, cost.detach(), reg=1, numItermax=50)
        # # 加权重建余弦距离
        # loss = torch.sum(T * cost)
        # print(loss)


        # generated_texts = self.processor.batch_decode(
        #     generated_ids,
        #     skip_special_tokens=True,
        # )

        # print(generated_texts[0])

# SmolVLM_256 = SmolVLM_256M_FeatureExtractor()
# SmolVLM_256.ceshi()