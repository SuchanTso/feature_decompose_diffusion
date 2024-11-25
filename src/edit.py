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
    ratio = 1.0, # let's define ratio as the percentage of how far we went from x_T . aka x_T , ... , x_1 , x_0
    eta = 0.0
    ):
    assert ratio <= 1.0 , "fool , no way to exceed the total step while diffusion"

    total_loops = diffusion.get_ddim_sample_total_loops()
    target_step = max(round(total_loops * ratio) - 1 , 0)
    latent_range = [target_step , total_loops]
    contineous_range = [0 , target_step]
    print("============gen_latent==============")

    latent , middle_noise = gen_latent_diffusion(
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

    manipulated_latent = manipulate_latent_space(laten_space=latent)

    manipulated_noise = diffusion.get_noised_img_from_middle(model , manipulated_latent , middle_noise , shape , target_step , clip_denoised , model_kwargs , eta , device)
    # edit image by manipulate latent in the bottle neck layer of unet , thus changing noise predicted

    final_noise_list,_,_ = diffusion.ddim_sample_loop(
        model,
        (batch_size, 3, args.image_size, args.image_size),
        noise=manipulated_noise,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        real_step=args.real_step,
        return_intermediate=True,  # 这里将 return_intermediate 显式设为 True
        step_range = contineous_range
    )
    final_noise = final_noise_list[-1]
    print(f"final_noise.shape = {final_noise.shape}")
    print("============done_latent==============")

    return final_noise


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
    middle_noise = middle_noise_list[-1].to(device)
    latent = diffusion.get_unet_middle_output(model , middle_noise , target_step , model_kwargs)

    return latent , middle_noise

def manipulate_latent_space(laten_space):
    #TODO: implete 
    #arg: latent_space = [h , hs , emb] , so do the h part
    # 展平特征矩阵
    features , Hs , emb = laten_space
    n , c , w , h = features.shape
    
    # Parameters
    num_components = 10  # Number of singular values/components to keep

    # Result container
    manipulated_features_list = []


    # Apply SVD independently to each image
    for i in range(features.shape[0]):  # Loop over the 4 images
        image_features = features[i].detach().cpu().numpy().astype(np.float32)  # Shape: [1024, 8, 8]
        
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
        # reduced_features = np.dot(U_reduced, S_reduced).T.reshape(num_components, 8, 8)  # Shape: [256, 8, 8]
        scale_factors = 0.5 #tem
        manipulated_S = np.copy(S)
        manipulated_S[:num_components] *= scale_factors 
        reconstructed_features = (U @ np.diag(manipulated_S) @ Vt).T  # Shape: [1024, 64]
        manipulated_features = reconstructed_features.reshape(image_features.shape)  # Shape: [1024, 8, 8]

        
        manipulated_features_list.append(manipulated_features)

        print(f"Original shape: {features.shape}")
        print(f"manipulated_features shape: {manipulated_features.shape}")

    manipulated_features_res = np.stack(manipulated_features_list , axis=0)
    manipulated_features_res = torch.tensor(manipulated_features_res).to(features.device , dtype=features.dtype)

    print(f"manipulated_features shape: {manipulated_features_res.shape}")
    assert manipulated_features_res.shape == features.shape

    return manipulated_features_res , Hs , emb