from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_for_reverse
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

from transformer import StaticAttention

def gen_featured_img(
    args,
    model,
    diffusion,
    shape,
    clip_denoised,
    batch_size,
    noise=None,
    model_kwargs=None,
    device = 'cpu',
    ratio = 1.0, # let's define ratio as the percentage of how far we went from x_0 . aka x_T , ... , x_1 , x_0
    eta = 0.0
    ):
    assert ratio <= 1.0 , "fool , no way to exceed the total step while diffusion"

    total_loops = diffusion.get_ddim_sample_total_loops()
    target_step = max(round(total_loops * ratio) - 1 , 0)
    latent_range = [target_step , total_loops]
    contineous_range = [0 , target_step]
    print("============gen_latent==============")
    print(f"stage1:[{total_loops},{target_step}]")
    print(f"stage2:[{target_step},0]")


    latent , middle_noise , contineous_noise = gen_latent_diffusion(
        args=args,
        model=model,
        diffusion=diffusion,
        batch_size=batch_size,
        noise=noise,
        model_kwargs=model_kwargs,
        device=device,
        ddim_range=latent_range,
        target_step=target_step
        )

    manipulated_latents = manipulate_latent_space(laten_space=latent)
    decomposed_img = []
    print("============gen_latent_with one s_value==============")
    non_edit_noise = diffusion.get_noised_img_from_middle(model , latent , middle_noise , shape , target_step , clip_denoised , model_kwargs , eta , device)

    for manipulated_latent in manipulated_latents:

        manipulated_noise = diffusion.get_noised_img_from_middle(model , manipulated_latent , middle_noise , shape , target_step , clip_denoised , model_kwargs , eta , device)
        # edit image by manipulate latent in the bottle neck layer of unet , thus changing noise predicted

        final_noise_list,_,_ = diffusion.ddim_sample_loop(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=contineous_noise + 10.0 * (manipulated_noise - non_edit_noise),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
            return_intermediate=True,  # 这里将 return_intermediate 显式设为 True
            step_range = contineous_range
        )
        final_noise = final_noise_list[-1]
        decomposed_img.append(final_noise)
        print("============done_latent==============")

    return decomposed_img


def gen_latent_diffusion(
        args,
        model,
        diffusion,
        batch_size,
        noise=None,
        model_kwargs=None,
        device = 'cpu',
        ddim_range = None,
        target_step = 1000
        ):
    assert ddim_range is not None 
    middle_noise_list,_,_ = diffusion.ddim_sample_loop(
        model,
        (batch_size, 3, args.image_size, args.image_size),
        noise=noise,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        real_step=args.real_step,
        return_intermediate=True,  # 这里将 return_intermediate 显式设为 True
        step_range = ddim_range
    )
    middle_noise = middle_noise_list[-2].to(device)
    continueous_noise = middle_noise_list[-1].to(device)
    latent = diffusion.get_unet_middle_output(model , middle_noise , target_step , None)

    return latent , middle_noise , continueous_noise

def manipulate_latent_space(laten_space):
    #TODO: implete 
    #arg: latent_space = [h , hs , emb] , so do the h part
    # 展平特征矩阵
    features , Hs , emb = laten_space
    # return laten_space
    n , c , w , h = features.shape
    
    # Parameters
    num_components = 5  # Number of singular values/components to keep
    latent_spaces = []
    for num_ in range(0 , num_components):
        hs_copy = list(Hs)
        feat_copy = features.clone()
        # for i in range(len(Hs)):
        #     hs_copy[i] = pca_feature(h_layers=Hs[i] , num_components=num_components , manipulate_fn=remain_one_comp , num_comp=num_)

        feat_copy = pca_feature(h_layers=features , num_components=num_components , manipulate_fn=remain_one_comp , num_comp=num_)
        # # Apply SVD independently to each image
        latent_spaces.append([feat_copy , hs_copy , emb])

    return latent_spaces

def svd_features(h_layers , num_components , manipulate_fn = None , num_comp = 0):
    manipulated_features_list = []
    for i in range(h_layers.shape[0]):  # Loop over the 4 images
        image_features = h_layers[i].detach().cpu().numpy().astype(np.float32)  # Shape: [1024, 8, 8]
        # Reshape each image feature map for SVD
        reshaped_image_features = image_features.reshape(image_features.shape[0], -1).T  # Shape: [64, 1024]
        
        # Apply SVD
        U, S, Vt = np.linalg.svd(reshaped_image_features, full_matrices=False)  # Shapes: [64, 64], [64], [64, 1024]
        print(f"check point1 : U.shape:{U.shape} , S:{S.shape} , Vt.shape:{Vt.shape}")

        # Truncate to the desired number of components
        U_reduced = U[:, :num_components]  # Shape: [64, 256]
        S_reduced = np.diag(S[:num_components])  # Shape: [256, 256]
        Vt_reduced = Vt[:num_components, :]  # Shape: [256, 1024]
        print(f"check point2 : U_reduced.shape:{U_reduced.shape} , S_reduced:{S_reduced.shape}")
        # Compute the reduced representation

        manipulated_S = np.copy(S)
        print(f"manipulate s_values = {manipulated_S}")

        if manipulate_fn:
            manipulated_S = manipulate_fn(manipulated_S , num_comp)

        print(f"after manipulate s_values = {manipulated_S}")
        reconstructed_features = (U @ np.diag(manipulated_S) @ Vt).T  # Shape: [1024, 64]
        manipulated_features = reconstructed_features.reshape(image_features.shape)  # Shape: [1024, 8, 8]

        
        manipulated_features_list.append(manipulated_features)

        print(f"Original shape: {h_layers.shape}")
        print(f"manipulated_features shape: {manipulated_features.shape}")

    manipulated_features_res = np.stack(manipulated_features_list , axis=0)

    manipulated_features_res = torch.tensor(manipulated_features_res).to(h_layers.device , dtype=h_layers.dtype)
    
    return manipulated_features_res

def manipulate_high(s_values):
    for s_value in s_values:
        s_value * 1.5
    return s_values

def manipulate_low(s_values):
    s_values[0] * 0.5
    return s_values

def remain_one_comp(s_values , num_index):
    remain = False
    for i in range(0 , len(s_values)):
        if i != num_index:
            s_values[i] = 0.0
        else:
            remain = True
    assert remain == True , f"num_index={num_index} is out of range s_values size"
    return s_values


def pca_feature(h_layers , num_components , manipulate_fn = None , num_comp = 0):
    
    manipualted_features_list = []
    for i in range(len(h_layers)):
        C, H, W = h_layers[i].shape
        print(f"h_layer = {h_layers[i].type()}")
        model = StaticAttention(C).to(h_layers.device)
        attentioned_feature = model(h_layers[i].to(torch.float))
        image_features = attentioned_feature.detach().cpu().numpy().astype(np.float32)  # Shape: [1024, 8, 8]
        flattened_features = image_features.reshape(C, -1).T  # 转换为 (H*W, C)

        # 1. 对瓶颈层特征进行 PCA
        pca = PCA(n_components=num_components)
        pca_result = pca.fit_transform(flattened_features)  # 主成分表示 (H*W, n_components)
        eigenvectors = pca.components_.T  # 特征方向矩阵 (C, n_components)
        mean_vector = pca.mean_  # 特征均值

        # 2. 重构瓶颈层特征，只保留第一个主成分
        component_idx = 0
        reconstructed_features = np.dot(pca_result[:, component_idx:component_idx+1], eigenvectors[:, component_idx:component_idx+1].T)
        reconstructed_features = reconstructed_features.T.reshape(C, H, W)
        manipualted_features_list.append(reconstructed_features)

    manipulated_features_res = np.stack(manipualted_features_list , axis=0)

    manipulated_features_res = torch.tensor(manipulated_features_res).to(h_layers.device , dtype=h_layers.dtype)
    
    return manipulated_features_res
"""
===========================================================================================================================================================
"""
def decompose_noised_image(diffusion , noised_images , device):
    overall_noised_image = gen_overall_noised_image(noised_images=noised_images , device=device)
    batch_size, channels,T, height, width = overall_noised_image.shape

    reshaped_features = overall_noised_image.permute(0, 3, 4, 1, 2)  # Flatten spatial dimensions
    reshaped_features = reshaped_features.reshape(-1, channels * T) # [batch_size, height * width, channels]

    from sklearn.cluster import KMeans

    components = []
    for i in range(batch_size):
        features = reshaped_features.cpu().numpy()
        num_components = 5#min(features.shape[0], features.shape[1])  # Dynamic component count
        kmeans = KMeans(n_clusters=num_components, random_state=42)
        labels = kmeans.fit_predict(features)

        # Group pixels into clusters
        clustered_image = labels.reshape(height, width)
        components.append(clustered_image)

    return components

def gen_overall_noised_image( noised_images , device):
    timestep = len(noised_images)
    high_noised_step = torch.stack(noised_images , dim = 2)
    print(f"high_noised_step.shape = {high_noised_step.shape}")
    return high_noised_step
    weights = torch.linspace(1.2, 0.01, steps=timestep).view(-1, 1, 1, 1).to(device)
    weights_norm = weights / weights.sum()
    weighted_tensors = [tensor * weight for tensor, weight in zip(noised_images, weights_norm)]
    weighted_mean_tensor = torch.stack(weighted_tensors).sum(dim=0)
    print("加权平均后的张量:", weighted_mean_tensor.shape)
    return weighted_mean_tensor




def visualize_components(diffusion , noised_image , device ,image_path):
    """
    Visualizes the decomposed components as segmentation masks.
    
    Args:
        components: List of 2D arrays (height x width), one for each decomposed component.
        original_image: Optional; Original image tensor [channels, height, width] for overlay.
    """
    components = decompose_noised_image(diffusion=diffusion , noised_images= noised_image , device=device)
    num_components = len(components)
    print(f"component.shape = {components[0].shape}")
    original_image = noised_image[0][0]
    print(f"original_image.shape = {original_image.shape}")
    fig, axes = plt.subplots(1, num_components + (1 if original_image is not None else 0), figsize=(15, 5))

    if original_image is not None:
        # Display the original image
        axes[0].imshow(original_image.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Original Image")
        axes[0].axis("off")

    # Display each component as a segmentation mask
    for i, component in enumerate(components):
        mask = component.astype(np.uint8)
        axes[i + (1 if original_image is not None else 0)].imshow(mask, cmap="tab20")  # Use a categorical colormap
        axes[i + (1 if original_image is not None else 0)].set_title(f"Component {i+1}")
        axes[i + (1 if original_image is not None else 0)].axis("off")

    plt.tight_layout()
    plt.savefig(image_path)

