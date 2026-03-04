import os
import sys
import argparse
import re
import requests
from PIL import Image
from typing import Dict, Any, List, Tuple
import hydra
import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm
# import wandb
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from google import genai
from openai import OpenAI
import anthropic

from utils import (
    get_api_key,
    hash_training_config,
    # setup_wandb,
    ensure_dir,
    encode_image,
    get_output_paths,
)

from vlmeval.config import supported_VLM

# Define valid image extensions
VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]


# def setup_gemini(api_key: str):
#     return genai.Client(api_key=api_key)

def setup_gemini(api_key: str):
    return OpenAI(
        api_key=api_key,
        base_url='https://api.vveai.com/v1'
    )


def setup_claude(api_key: str):
    return anthropic.Anthropic(api_key=api_key, base_url='https://api.vveai.com')


# def setup_gpt4o(api_key: str):
#     return OpenAI(
#         api_key="api_key",
#     )
def setup_gpt4o(api_key: str):
    return OpenAI(
        api_key=api_key,
        base_url='https://api.vveai.com/v1'
        # base_url='https://api.v36.cm/v1'
    )


def setup_gpt5(api_key: str):
    return OpenAI(
        api_key=api_key,
        # base_url='https://api.vveai.com/v1'
        base_url='https://api.v36.cm/v1'
    )


def get_media_type(image_path: str) -> str:
    """Get the correct media type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


class ImageDescriptionGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Get API key for the model
        api_key = get_api_key(model_name)

        if model_name == "gemini":
            self.client = setup_gemini(api_key)
        elif model_name == "claude":
            self.client = setup_claude(api_key)
        elif model_name == "claude4_5":
            self.client = setup_claude(api_key)
        elif model_name == "gpt4o":
            self.client = setup_gpt4o(api_key)
        elif model_name == "gpt5":
            self.client = setup_gpt5(api_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate_description(self, image_path: str) -> str:
        if self.model_name == "gemini":
            return self._generate_gemini(image_path)
        elif self.model_name == "claude":
            return self._generate_claude(image_path)
        elif self.model_name == "claude4_5":
            return self._generate_claude4_5(image_path)
        elif self.model_name == "gpt4o":
            return self._generate_gpt4o(image_path)
        elif self.model_name == "gpt5":
            return self._generate_gpt5(image_path)

    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    # def _generate_gemini(self, image_path: str) -> str:
    #     image = Image.open(image_path)
    #     response = self.client.models.generate_content(
    #         model="gemini-2.0-flash",
    #         contents=["Describe this image, no longer than 20 words.", image],
    #     )
    #     return response.text.strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gemini(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
            temperature=0,
        )
        # print(response)
        # return response.choices[0].message.content
        content = response.choices[0].message.content
        return content.replace("\n", " ").strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )
        return response.content[0].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude4_5(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=300,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )
        return response.content[0].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gpt4o(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        # print(response)
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gpt5(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]
        )
        return response.output_text


def save_descriptions(descriptions: List[Tuple[str, str]], output_file: str):
    """Save image descriptions to file."""
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        for filename, desc in descriptions:
            f.write(f"{filename}: {desc}\n")


def generate_descriptions(args):
    output_dir = os.path.join(args.output, "img")
    desc_output_adv = os.path.join(args.output, "description")
    ensure_dir(desc_output_adv)

    desc_output_src_tgt = args.des_path_src_tgt
    ensure_dir(desc_output_src_tgt)

    target_txt_path = os.path.join(desc_output_src_tgt, f"target_{args.model_name}.txt")
    source_txt_path = os.path.join(desc_output_src_tgt, f"source_{args.model_name}.txt")
    adv_txt_path = os.path.join(desc_output_adv, f"adversarial_{args.model_name}.txt")

    print("Using model:", args.model_name)
    print("Processing images...")

    # ============================================================
    # 判断是否需要生成 target / source
    # ============================================================

    generate_target = not os.path.exists(target_txt_path)
    generate_source = not os.path.exists(source_txt_path)

    if not generate_target:
        print("Target description file exists. Skip generating target.")
    if not generate_source:
        print("Source description file exists. Skip generating source.")

    # ============================================================
    # 判断模型类型（闭源 or 开源）
    # ============================================================

    closed_source_models = ["gpt4o", "gpt5", "gemini", "claude", "claude4_5"]
    open_source_models = [
        "Qwen2.5-VL-7B-Instruct",
        "InternVL3-8B",
        "llava_v1.5_7b",
        "glm-4.1v-9b-thinking"
    ]

    if args.model_name in closed_source_models:
        model_type = "closed"
        generator = ImageDescriptionGenerator(model_name=args.model_name)

    elif args.model_name in open_source_models:
        model_type = "open"
        try:
            model = supported_VLM[args.model_name]()
        except KeyError:
            raise ValueError(f"Model {args.model_name} not supported.")
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    # ============================================================
    # 开始生成
    # ============================================================

    tgt_descriptions = []
    src_descriptions = []
    adv_descriptions = []

    for root, _, files in os.walk(output_dir):

        for file in tqdm(files):

            if not any(file.lower().endswith(ext.lower())
                       for ext in VALID_IMAGE_EXTENSIONS):
                continue

            try:
                adv_path = os.path.join(root, file)
                filename_base = os.path.splitext(file)[0]

                # -------------------------
                # 找 target 图
                # -------------------------
                tgt_path = None
                for ext in VALID_IMAGE_EXTENSIONS:
                    candidate = os.path.join(
                        args.tgt_data_path,
                        "1",
                        filename_base + ext
                    )
                    if os.path.exists(candidate):
                        tgt_path = candidate
                        break

                if tgt_path is None:
                    print(f"Target not found for {filename_base}, skip.")
                    continue

                # -------------------------
                # 找 source 图
                # -------------------------
                src_path = None
                for ext in VALID_IMAGE_EXTENSIONS:
                    candidate = os.path.join(
                        args.src_data_path,
                        "nips17",
                        filename_base + ext
                    )
                    if os.path.exists(candidate):
                        src_path = candidate
                        break

                if src_path is None:
                    print(f"Source not found for {filename_base}, skip.")
                    continue

                print("ADV Path:", adv_path)

                prompt = "Describe this image in one concise sentence, no longer than 20 words."

                # ============================================================
                # 生成函数统一封装
                # ============================================================

                def generate_desc(image_path):
                    if model_type == "closed":
                        return generator.generate_description(image_path)
                    elif model_type == "open":
                        desc = model.generate([image_path, prompt])

                        # 特殊处理 GLM
                        if args.model_name == "glm-4.1v-9b-thinking":
                            match = re.search(r"<answer>(.*?)</answer>", desc, re.DOTALL)
                            if match:
                                desc = match.group(1).strip()
                        return desc

                # -------------------------
                # Target
                # -------------------------
                if generate_target:
                    tgt_desc = generate_desc(tgt_path)
                    tgt_descriptions.append((file, tgt_desc))
                    print("TGT:", tgt_desc)

                # -------------------------
                # Source
                # -------------------------
                if generate_source:
                    src_desc = generate_desc(src_path)
                    src_descriptions.append((file, src_desc))
                    print("SRC:", src_desc)

                # -------------------------
                # Adversarial（始终生成）
                # -------------------------
                adv_desc = generate_desc(adv_path)
                adv_descriptions.append((file, adv_desc))
                print("ADV:", adv_desc)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    # ============================================================
    # 保存结果
    # ============================================================

    if generate_target:
        save_descriptions(tgt_descriptions, target_txt_path)
        print("Target descriptions saved.")

    if generate_source:
        save_descriptions(src_descriptions, source_txt_path)
        print("Source descriptions saved.")

    save_descriptions(adv_descriptions, adv_txt_path)
    print("Adversarial descriptions saved.")

    print("\nDescriptions finished.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./MPCAttack")
    parser.add_argument("--src_data_path", type=str, default="resources/images/bigscale")
    parser.add_argument("--tgt_data_path", type=str, default="resources/images/target_images")
    parser.add_argument("--des_path_src_tgt", type=str, default="descriptions_src_tgt")
    parser.add_argument("--model_name", type=str, default="gpt4o",
                        help="closed: gpt4o, gpt5, gemini, claude, claude4_5"
                             "open: Qwen2.5-VL-7B-Instruct, InternVL3-8B, llava_v1.5_7b, glm-4.1v-9b-thinking")
    return parser.parse_args()


# ============================================================
# Entry
# ============================================================

def main():
    args = get_args()
    generate_descriptions(args)


if __name__ == "__main__":
    main()
