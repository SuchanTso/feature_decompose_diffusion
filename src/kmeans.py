import numpy as np
from sklearn.cluster import KMeans
import torch
from PIL import Image
import matplotlib.pyplot as plt


def kmeans_approach(image , cluster_num , device):
    # 转换为 PyTorch 张量，调整为 [B, C, H, W] 格式
    image_tensor = ((torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0))).to(device)
    batch_size ,c ,  width , height =image_tensor.shape
    print(f"image_tensor.shape = {image_tensor.shape}")
    # 将图像展平为二维数组 (batch_size * width * height, 3)
    flattened_images = image_tensor.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
    # 定义聚类的簇数

    # 使用 K-Means 聚类
    kmeans = KMeans(n_clusters=cluster_num, random_state=42)
    kmeans.fit(flattened_images)

    # 获取每个像素的聚类标签
    cluster_labels = kmeans.labels_

    # 将标签重新整理为原始图像的形状 (batch_size, width, height)
    clustered_images = cluster_labels.reshape(batch_size, width, height)

    # 可视化结果示例
    print("聚类后的结果形状:", clustered_images.shape)

    return clustered_images

#=================test code===========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_path ="C:/my/Code/projects/feature_decompose_diffusion/datasets/flower"
    bed_path = "/data/usr/zsc/project/feature_decompose_diffusion/datasets/flowers"
    img_path = f"{bed_path}/2521408074_e6f86daf21_n.jpg"

    image = Image.open(img_path).convert("RGB").resize((256,256))
    # 转换为 numpy 数组并归一化到 [-1, 1]
    image = np.array(image) / 127.5 - 1.0
    clustered_images = kmeans_approach(image , 7 , device)
    # 可视化第一张图像的聚类结果
    plt.imshow(clustered_images[0], cmap='viridis')
    plt.title("Clustered Image")
    plt.colorbar()
    plt.savefig(f"kmeans.jpg")
#=================test code===========================================
