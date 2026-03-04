import os
import argparse
import random
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm
from PIL import Image

from surrogates import (
    DINOv2FeatureExtractor,
    InternVL3_1B_FeatureExtractor,
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureExtractor_our,
)

from utils import ensure_dir, info_nce_loss

CLIP_BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(pic):
    img = torch.from_numpy(np.array(pic, np.uint8, copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.float()


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, label, path


def build_clip_models(args):
    if not args.ensemble and len(args.clip_backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone allowed")

    models = []
    for name in args.clip_backbone:
        if name not in CLIP_BACKBONE_MAP:
            raise ValueError(f"Unknown clip_backbone: {name}")

        model = CLIP_BACKBONE_MAP[name]().eval().to(args.device)
        model.requires_grad_(False)
        models.append(model)

    if args.ensemble:
        extractor = EnsembleFeatureExtractor_our(models)
    else:
        extractor = models[0]

    return extractor, models



def extract_feature(img, clip_models, internvl_model, dino_model):
    clip_feats = clip_models(img)
    clip_feats = torch.cat(list(clip_feats.values()), dim=1)

    intern_feat = internvl_model(img).mean(dim=1)
    dino_feat = dino_model(img)

    z = torch.cat([clip_feats, intern_feat, dino_feat], dim=1)
    return z



def run_attack(args):
    device = args.device

    transform_fn = transforms.Compose([
        transforms.Resize(args.input_res,
                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_res),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(lambda img: to_tensor(img)),
    ])

    # input transformation
    source_crop = (transforms.RandomResizedCrop(args.input_res, scale=args.crop_scale))

    source_dataset = ImageFolderWithPaths(args.cle_data_path, transform=transform_fn)
    target_dataset = ImageFolderWithPaths(args.tgt_data_path, transform=transform_fn)

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    # Loading models
    clip_models, _ = build_clip_models(args)
    internvl_model = InternVL3_1B_FeatureExtractor().eval().to(device)
    internvl_model.requires_grad_(False)
    dino_model = DINOv2FeatureExtractor().eval().to(device)
    dino_model.requires_grad_(False)

    for i, ((image_org, _, path_org), (image_tgt, _, path_tgt)) in enumerate(zip(source_loader, target_loader)):
        print(i, path_org, path_tgt)

        src_text = internvl_model.generate_description(image_org)
        tar_text = internvl_model.generate_description(image_tgt)
        print(src_text)
        print(tar_text)

        # Initialize perturbation
        delta = 16 / 255 * torch.randn_like(image_org).to(device)
        delta.requires_grad_(True)

        img_src = image_org.to(device)
        img_tgt = image_tgt.to(device)

        with torch.no_grad():
            text_feature_src = clip_models.get_text_features(src_text)  # list
            text_feature_tar = clip_models.get_text_features(tar_text)

            src_teats = clip_models(img_src)  # list [1, 512]
            tgt_teats = clip_models(img_tgt)

            src_feat_internvl = internvl_model(img_src)  # torch.Size([1, 256, 896])
            tgt_feat_internvl = internvl_model(img_tgt)

            src_feat_dino = dino_model(img_src)  # torch.Size([1, 768])
            tgt_feat_dino = dino_model(img_tgt)

            clip_src = []
            clip_tgt = []

            for index in range(len(src_teats)):
                src_feat_clip = args.lam * src_teats[index] + (1 - args.lam) * text_feature_src[index]
                tgt_feat_clip = args.lam * tgt_teats[index] + (1 - args.lam) * text_feature_tar[index]
                clip_src.append(src_feat_clip)
                clip_tgt.append(tgt_feat_clip)
            clip_src = torch.cat(clip_src, dim=1)
            clip_tgt = torch.cat(clip_tgt, dim=1)

            internvl_src = src_feat_internvl.mean(dim=1).float()
            dino_src = src_feat_dino
            z_src = torch.cat([clip_src, internvl_src, dino_src], dim=1)

            internvl_tgt = tgt_feat_internvl.mean(dim=1).float()
            dino_tgt = tgt_feat_dino
            z_tgt = torch.cat([clip_tgt, internvl_tgt, dino_tgt], dim=1)


        momentum = torch.zeros_like(delta, requires_grad=False)
        pbar = tqdm(range(300), desc=f"Attack Progress")
        for epoch in pbar:

            adv_img = img_src + delta
            local_cropped = source_crop(adv_img)

            adv_feats = clip_models(local_cropped)
            adv_feat_internvl = internvl_model(local_cropped)
            adv_feat_dino = dino_model(local_cropped)

            clip_adv = torch.cat(list(adv_feats.values()), dim=1)
            internvl_adv = adv_feat_internvl.mean(dim=1).float()
            dino_adv = adv_feat_dino
            z_adv = torch.cat([clip_adv, internvl_adv, dino_adv], dim=1)
            z_sim_src = torch.mean(F.cosine_similarity(z_adv, z_src.detach(), dim=-1))
            z_sim_tgt = torch.mean(F.cosine_similarity(z_adv, z_tgt.detach(), dim=-1))

            loss = info_nce_loss(z_adv, z_tgt.detach(), z_src.detach(), temperature=args.tau, omega=args.omega)  # infonce loss


            if epoch % 10 == 0:
                print(f'epoch:{epoch}', f'loss:{loss.item()}', f'z_sim_src:{z_sim_src.item()}', f'z_sim_tgt:{z_sim_tgt.item()}')


            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

            # MI-FGSM update
            momentum = momentum * 0.9 + grad
            delta.data = torch.clamp(
                delta + args.alpha * torch.sign(momentum),
                min=-args.epsilon,
                max=args.epsilon,
            )
            # torchvision.utils.save_image(delta.detach().cpu(), f"delta1/delta_{epoch}.png")

        # Create final adversarial image
        adv_image = image_org + delta.detach().cpu()
        adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

        adv_text = internvl_model.generate_description(adv_image * 255)
        print(tar_text)
        print(adv_text)
        # sys.exit()


        # Save images
        for path_idx in range(len(path_org)):
            folder, name = (
                os.path.basename(os.path.dirname(path_org[path_idx])),
                os.path.basename(path_org[path_idx]),
            )
            folder_to_save = os.path.join(args.output, "img", folder)
            ensure_dir(folder_to_save)

            if "JPEG" in name:
                torchvision.utils.save_image(
                    adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + "png"
                )
            elif "png" in name:
                torchvision.utils.save_image(
                    adv_image[path_idx], os.path.join(folder_to_save, name)
                )




def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--cle_data_path", type=str, default="resources/images/bigscale")
    parser.add_argument("--tgt_data_path", type=str, default="resources/images/target_images")
    parser.add_argument("--output", type=str, default="./MPCAttack")
    parser.add_argument("--batch_size", type=int, default=1)

    # Model
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--crop_scale", nargs=2, type=float, default=[0.5, 0.9])
    parser.add_argument("--ensemble", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--clip_backbone", nargs="+", default=["B16", "B32", "Laion"])

    # Optim
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=16)
    parser.add_argument("--steps", type=int, default=300)

    # MPCAttack Parameter
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--omega", type=float, default=2.0)

    return parser.parse_args()


def main():
    args = get_args()
    set_seed(2023)
    run_attack(args)


if __name__ == "__main__":
    main()
