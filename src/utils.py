import argparse
import os
import torch
import sys
import cv2
#from mpi4py import MPI
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th
import torch.distributed as dist

from torchvision.utils import make_grid
from PIL import Image
import torchvision

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_for_reverse
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs

def get_model_ready(args , device):
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    
    # Load model state dict directly specifying device mapping
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Move model to the specified device
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion


def create_argparser():
    defaults = dict(
        images_dir="/data/usr/testuser/code/dataset/SDXL-face-tmp/SDXL-fake/",
        recons_dir="/data2/wangzd/dataset/DiffusionForensics/recons",
        dire_dir="/data/usr/zsc/project/guided-diffusion/imgs/dire",
        clip_denoised=True,
        num_samples=-1,
        batch_size=16,
        use_ddim=False,
        model_path="",
        real_step=1,#如果real_step为0则会加num_timesteps步噪声，所以噪声加多了，效果差；如果设置成一个非零的小整数T，则只会加T步噪声，去除T步噪声，所以噪声少，去噪好。
        continue_reverse=False,
        has_subfolder=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def visualize_noise(model , diffusion , noise_list , dir):
    total_loops = diffusion.get_ddim_sample_total_loops()

    for i , noise in enumerate(reversed(noise_list)):
        t = total_loops - i - 1
        latent , hs , emb = diffusion.get_unet_middle_output(model , noise , t , None)
        save_path = f"{dir}/latent_{t}.jpg"
        pca_latent_visual(latent , noise.cpu() , save_path)


def pca_latent_visual(latent , nosie , save_path):
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt


    # 假设 `tensor_data` 是形状为 (batch_size, 1024, 8, 8) 的张量

    # 展平数据 (batch_size, 1024, 8, 8) -> (batch_size, 1024 * 8 * 8)
    batch_size , channal , w , h = latent.shape
    flattened_data = latent[0].view(channal, -1).cpu().detach().numpy()

    # 使用 PCA 将数据降维到 3D
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(flattened_data)  # 输出形状为 (batch_size, 3)

    fig = plt.figure(figsize=(15, 5))

    # 第一列：绘制噪声图（示例用第一个样本的一个通道）
    ax1 = fig.add_subplot(121)
    
    sample_noise = ((nosie[0].clamp(-1 , 1) + 1) / 2 * 255.0).type(torch.uint8).permute(1, 2, 0)  # 取第一个样本的第一个通道
    ax1.imshow(sample_noise, cmap='gray')
    ax1.set_title('Sample Noise (First Channel)')
    ax1.axis('off')

    # 第二列：绘制 3D PCA 可视化
    ax2 = fig.add_subplot(122, projection='3d')
    colors = np.repeat(np.arange(batch_size), reduced_data.shape[0] // batch_size)

    sc = ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
                    c=colors, cmap='viridis', s=50)
    ax2.set_title('3D PCA Visualization')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_zlabel('Principal Component 3')

    # 添加颜色条
    plt.colorbar(sc, ax=ax2, label='Sample Index')

    plt.tight_layout()
    plt.savefig(save_path)
