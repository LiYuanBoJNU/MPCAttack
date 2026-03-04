import torch
import torch.nn.functional as F

PATCH_SIZE   = 512          # 每块大小
TARGET_SHAPE = (4 * PATCH_SIZE, 4 * PATCH_SIZE)  # 2048×2048
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1)
STD  = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1)

def resize_to_2048_square(img: torch.Tensor) -> torch.Tensor:
    """
    把 img(C,H,W) 无黑边地 resize 成 2048×2048
    先用双线性插值，再中心裁剪/填充到 2048×2048
    """
    C, H, W = img.shape
    # 等比例缩放，使得短边≥2048
    scale = max(TARGET_SHAPE[0]/H, TARGET_SHAPE[1]/W)
    new_H, new_W = int(H*scale), int(W*scale)
    # 保证偶数
    new_H += new_H % 2
    new_W += new_W % 2
    resized = F.interpolate(
        img.unsqueeze(0), size=(new_H, new_W),
        mode='bilinear', align_corners=False
    ).squeeze(0)  # [C,new_H,new_W]

    # 中心裁剪/填充到 2048×2048
    pad_h = max(TARGET_SHAPE[0] - new_H, 0)
    pad_w = max(TARGET_SHAPE[1] - new_W, 0)
    resized = F.pad(resized, (0, pad_w, 0, pad_h))
    # 中心裁剪
    _, H2, W2 = resized.shape
    y0 = (H2 - TARGET_SHAPE[0]) // 2
    x0 = (W2 - TARGET_SHAPE[1]) // 2
    return resized[:, y0:y0+TARGET_SHAPE[0], x0:x0+TARGET_SHAPE[1]]

def split_into_17(img: torch.Tensor) -> torch.Tensor:
    """
    img: [3,H,W] 任意分辨率
    return: [1,17,3,512,512]
    """
    img = resize_to_2048_square(img)  # [3,2048,2048]

    # 切成 4×4 patch
    patches = []
    for y in range(0, 2048, PATCH_SIZE):
        for x in range(0, 2048, PATCH_SIZE):
            patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patches.append(patch)
    # 第 0 张为缩略图
    thumbnail = F.interpolate(
        img.unsqueeze(0), size=(PATCH_SIZE, PATCH_SIZE),
        mode='bilinear', align_corners=False
    ).squeeze(0)
    patches.insert(0, thumbnail)

    out = torch.stack(patches, dim=0)      # [17,3,512,512]
    out = (out - MEAN) / STD
    return out.unsqueeze(0)                # [1,17,3,512,512]

# ------------------ 测试 ------------------
if __name__ == "__main__":
    dummy = torch.rand(1, 3, 224, 224, requires_grad=True)
    output = split_into_17(dummy[0])
    print("输出形状:", output.shape)          # [1,17,3,512,512]

    loss = output.sum()
    loss.backward()
    print("梯度形状:", dummy.grad.shape)      # [1,3,720,1280]