from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch
import torch.nn as nn

# image_path = "/data/usr/zsc/project/feature_decompose_diffusion/datasets/lsun_bedroom/00000089629ce3ba87bae003073896ba01988dee.jpg"
# image = Image.open(image_path)

# processor = ViTImageProcessor.from_pretrained('../model/vit')
# model = ViTModel.from_pretrained('../model/vit')
# inputs = processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
# print(f"output.shape = {outputs} , state = {last_hidden_states.shape}")


class StaticAttention(nn.Module):
    def __init__(self, channels):
        super(StaticAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，用于通道注意力
        self.spatial_attention = nn.Conv2d(channels, 1, kernel_size=1)  # 空间注意力生成器

    def forward(self, x):
        # 通道注意力
        channel_weights = self.global_pool(x)  # (B, C, 1, 1)
        channel_weights = channel_weights / (channel_weights.sum(dim=1, keepdim=True) + 1e-6)  # 归一化
        x = x * channel_weights  # 通道加权

        # 空间注意力
        spatial_weights = torch.sigmoid(self.spatial_attention(x))  # (B, 1, H, W)
        x = x * spatial_weights  # 空间加权
        return x
