from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from pca import pca_analyse, nmf_analyse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dir_path ="C:/my/Code/projects/feature_decompose_diffusion/datasets/flower"
bed_path = "/data/usr/zsc/project/feature_decompose_diffusion/datasets/flowers"
img_path = f"{dir_path}/2521408074_e6f86daf21_n.jpg"

image = Image.open(img_path).convert("RGB").resize((256,256))
# 转换为 numpy 数组并归一化到 [-1, 1]
image = np.array(image) / 127.5 - 1.0
# 转换为 PyTorch 张量，调整为 [B, C, H, W] 格式
image_tensor = ((torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0))).to(device)


vae_path = "C:/my/Code/projects/feature_decompose_diffusion/model/vae"

vae = vae = AutoencoderKL.from_pretrained(vae_path)
vae.eval()  # 切换到评估模式

# 检查设备（CPU 或 GPU）
vae.to(device)
with torch.no_grad():
    latent_vector = vae.encode(image_tensor).latent_dist.sample()* 0.18215
    print("潜在空间的形状:", latent_vector.shape)  # [B, C, H, W]
    images = pca_analyse(latent_vector.cpu() , 3 , device ,vae)

    # images = nmf_analyse(latent_vector.cpu() , 3 , device ,vae)

    # reconstructed_image = vae.decode(components[0].to(device).to(vae.dtype)).sample#primary comp for temp

# 反归一化到 [0, 1]，然后转换为 numpy 格式
# reconstructed_image = ((reconstructed_image + 1) / 2).clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()

origin_image = ((image_tensor + 1) / 2).clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, len(images) + 1, figsize=(15, 5))
axes[0].imshow(origin_image, cmap='gray')
axes[0].set_title(f"origin")
axes[0].axis('off')
for i, img in enumerate(images):
    axes[i + 1].imshow(img)
    axes[i + 1].axis("off")
    axes[i + 1].set_title(f"Decoded Component {i+1}")
plt.tight_layout()
plt.savefig(f"./test.jpg")