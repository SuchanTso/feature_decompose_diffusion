import torch
from utils import*
import os
class Diffusion_processor:
    def __init__(self , args):
        self.forward_fn = args.diffusion.ddim_reverse_sample_loop#make img noise
        self.inverse_fn = args.diffusion.ddim_sample_loop#denoise process

    def diffusion_forward(self ,args,batch_size , imgs , model_kwargs):
        return self.forward_fn(
            args.model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )
    
    def diffusion_inverse(self, args , batch_size , noise ,record_ratio, model_kwargs):
        return self.inverse_fn(
            args.model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
            return_intermediate=False , # 这里将 return_intermediate 显式设为 True
            record_ratio=record_ratio
        )
    
    def save_image(self , image , save_path):
        # a)norm image 
        norm_img = ((image.clamp(-1 , 1) + 1) / 2 * 255.0).type(torch.uint8)
        # b)prepare directions recursively
        check_prepare_path(save_path)
        # c)save image
        save_images(norm_img , save_path)

    def data_update(self , args , batch_size):
        imgs, out_dicts, paths = next(args.img_data_loader)
        imgs = imgs[:batch_size]
        paths = paths[:batch_size]
        imgs = reshape_image(imgs, args.image_size)
        imgs = imgs.to(args.device)
        fn_save = os.path.basename(paths[-1])
        return imgs , paths , fn_save
    pass