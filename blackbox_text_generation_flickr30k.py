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

# https://api.v36.cm/
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
    )


def setup_gpt5(api_key: str):
    return OpenAI(
        api_key=api_key,
        base_url='https://api.vveai.com/v1'
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
    adv_txt_path = os.path.join(desc_output_adv, f"adversarial_{args.model_name}.txt")

    print("Using model:", args.model_name)
    print("Processing images...")

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



    adv_descriptions = []

    for root, _, files in os.walk(output_dir):

        for file in tqdm(files):

            if not any(file.lower().endswith(ext.lower())
                       for ext in VALID_IMAGE_EXTENSIONS):
                continue

            try:
                adv_path = os.path.join(root, file)
                print("ADV Path:", adv_path)
                prompt = "Describe this image in one concise sentence, no longer than 20 words."



                def generate_desc(image_path):
                    if model_type == "closed":
                        return generator.generate_description(image_path)
                    elif model_type == "open":
                        desc = model.generate([image_path, prompt])


                        if args.model_name == "glm-4.1v-9b-thinking":
                            match = re.search(r"<answer>(.*?)</answer>", desc, re.DOTALL)
                            if match:
                                desc = match.group(1).strip()
                        return desc

                adv_desc = generate_desc(adv_path)
                adv_descriptions.append((file, adv_desc))
                print("ADV:", adv_desc)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    save_descriptions(adv_descriptions, adv_txt_path)
    print("Adversarial descriptions saved.")

    print("\nDescriptions finished.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./MPCAttack_flickr30k")
    parser.add_argument("--model_name", type=str, default="gpt4o",
                        help="closed: gpt4o, gpt5, gemini, claude, claude4_5"
                             "open: Qwen2.5-VL-7B-Instruct, InternVL3-8B, llava_v1.5_7b, glm-4.1v-9b-thinking")
    return parser.parse_args()



def main():
    args = get_args()
    generate_descriptions(args)


if __name__ == "__main__":
    main()
