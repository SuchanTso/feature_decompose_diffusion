import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os
import torch
from sklearn.decomposition import NMF


# Load the RGB image for PCA
# dir_path = "/data/usr/zsc/project/feature_decompose_diffusion/datasets/flowers"
# folder_contents = os.listdir(dir_path)
# file_names = [os.path.splitext(item)[0] for item in folder_contents]
# save_path = f"/data/usr/zsc/project/feature_decompose_diffusion/imgs/analysis"
# os.makedirs( save_path, exist_ok=True)

# for item in file_names:
def pca_analyse(latent , components_num , device , vae):
    batch_size ,c , w, h = latent.shape
    # latent = (latent > (latent.max() / 2.0)).float()
    flattened = latent.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()  # [H*W, C]
    print(f"falttened shape = {flattened.shape}")
    # Perform PCA with 3 components
    pca = PCA(n_components=components_num)
    pca_result = pca.fit_transform(flattened)

    reconstructed = pca.inverse_transform(pca_result)

    components = pca.components_  # 主成分矩阵

    decoded_images = []
    for i in range(components_num):
        # 只保留单个主成分
        single_component = np.outer(pca_result[:, i], components[i])  # [H*W, C]

        # single_component = pca_result[:, i:i+1] @ components[i:i+1, :]
        reconstructed_latent =  single_component

        # 恢复到潜在空间形状
        reconstructed_tensor = (
            torch.tensor(reconstructed_latent, dtype=torch.float32)
            .reshape(batch_size, h, w, c)
            .permute(0, 3, 1, 2)
            .to(device)
        ).to(vae.dtype) / 0.18215

        print(f"reconstructed_tensor.shape = {reconstructed_tensor.shape}")
        
        with torch.no_grad():
            decoded_image_tensor = vae.decode(reconstructed_tensor).sample
            img = ((decoded_image_tensor + 1) / 2).clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
            decoded_image  = (img * 255).astype(np.uint8)
            decoded_images.append(decoded_image)

    components_images = [pca_result[:, i].reshape(h, w) for i in range(3)]
    components_images_normalized = [
    (comp - np.min(comp)) / (np.max(comp) - np.min(comp)) for comp in components_images
]

    # Visualize the first 3 PCA components in separate grayscale plots
    fig, axes = plt.subplots(1, components_num, figsize=(15, 5))
    for i, ax in enumerate(axes):
        T = 0.5 * components_images_normalized[i].max()  # 设定为最大值的一半
        # # 二值化
        binary_img = np.where(components_images_normalized[i] >= T, 255, 0).astype(np.uint8)
        # ax.imshow(components_images_normalized[i], cmap='gray')
        ax.imshow(binary_img, cmap='gray')

        ax.set_title(f"PCA Component {i + 1}")
        ax.axis('off')

    
    plt.tight_layout()
    plt.savefig(f"./comp_1.jpg")
    return decoded_images


def nmf_analyse(latent , components_num , device , vae):

    batch_size, c, h, w = latent.shape

    # 展平潜在表示为二维矩阵 [H*W, C]
    latent_flattened = latent.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
    # latent_min_value = latent_flattened.min()
    # latent_flattened -= latent_min_value
    latent_flattened = abs(latent_flattened)
    # 执行 NMF 分解
    nmf = NMF(n_components=components_num, init=None,  # W H 的初始化方法，包括'random' | 'nndsvd'(默认) |  'nndsvda' | 'nndsvdar' | 'custom'.
          solver='cd',  # 'cd' | 'mu'
          beta_loss='frobenius',  # {'frobenius', 'kullback-leibler', 'itakura-saito'}，一般默认就好
          tol=1e-4,  # 停止迭代的极限条件
          max_iter=200,  # 最大迭代次数
          random_state=None,
          l1_ratio=0.,  # 正则化参数
          verbose=0,  # 冗长模式
          shuffle=False  )# 针对"cd solver"
    W = nmf.fit_transform(latent_flattened)  # [H*W, n_components]
    H = nmf.components_  # [n_components, C]

    # 解码每个主成分
    decoded_images = []
    for i in range(components_num):
        # 构造单个主成分的潜在表示
        single_component = np.dot(W[:, i:i+1], H[i:i+1, :])  # [H*W, C]
        # single_component += latent_min_value  # 还原偏移

        reconstructed_latent = (
            torch.tensor(single_component, dtype=torch.float32)
            .reshape(batch_size, h, w, c)
            .permute(0, 3, 1, 2)  # 恢复到 [B, C, H, W]
            .to(device)
            .to(vae.dtype) / 0.18215  # VAE 通常需要调整尺度
        )

        # 使用 VAE 解码
        with torch.no_grad():
            decoded_tensor = vae.decode(reconstructed_latent).sample  # [B, 3, H, W]
            img = ((decoded_tensor + 1) / 2).clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
            decoded_image = (img * 255).astype(np.uint8)
            decoded_images.append(decoded_image)

    # 保存解码后的主成分图像
    fig, axes = plt.subplots(1, components_num, figsize=(15, 5))
    for i, image in enumerate(decoded_images):
        axes[i].imshow(image)
        axes[i].set_title(f"Decoded Component {i + 1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("decoded_components.png")
    print("解码后的主成分图像已保存为: decoded_components.png")

    return decoded_images