from utils import*
from edit import*
import time

def main():
    args = create_argparser().parse_args()
    logger.configure(dir=args.recons_dir)

    os.makedirs(args.recons_dir, exist_ok=True)

    # Directly specify the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, diffusion = get_model_ready(args=args , device=device)
    data = load_data_for_reverse(
        data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond
    )
    logger.log("computing recons & DIRE ...")
    have_finished_images = 0
    while have_finished_images < args.num_samples:
        # 直接使用args.batch_size，不需要基于MPI调整批次大小
        batch_size = min(args.batch_size, args.num_samples - have_finished_images)
        imgs, out_dicts, paths = next(data)
        imgs = imgs[:batch_size]
        paths = paths[:batch_size]
        imgs = imgs.to(device)
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=device)
            model_kwargs["y"] = classes

        reverse_fn = diffusion.ddim_reverse_sample_loop
        sample_fn = diffusion.ddim_sample_loop
        imgs = reshape_image(imgs, args.image_size)

        # noise origin image x_0 to x_T
        time1 = time.time()
        latent , noised_list= reverse_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )
        logger.log("done noise origin image...")

        time2 = time.time()
        # manipulate the latent space in "ratio" part of x_T and do the rest of ddim to generate
        de_imgs = gen_featured_img(args=args,
                               model=model,
                               diffusion=diffusion,
                               batch_size=batch_size,
                               shape=(batch_size, 3, args.image_size, args.image_size),
                               clip_denoised=args.clip_denoised,
                               noise=latent,
                               model_kwargs=model_kwargs,
                               device=device,
                               ratio=0.6
                               )

        origin_sample, eps_list, adjusted_eps_list= sample_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=latent,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
            return_intermediate=True , # 这里将 return_intermediate 显式设为 True
        )
        time3 = time.time()
        print(f"forward time = {time2 - time1} , reverse time = {time3 - time2} , total = {time3 - time1}")
            # 获取原始图像的文件夹和文件名
        original_dir_name = os.path.basename(os.path.dirname(paths[-1]))
        fn_save = os.path.basename(paths[-1])
    
        # 创建以原始图像命名的新文件夹
        recons_save_dir = os.path.join(args.recons_dir, original_dir_name, os.path.splitext(fn_save)[0])
        os.makedirs(recons_save_dir, exist_ok=True)

        input_filename = f"{os.path.splitext(fn_save)[0]}_input.png"
        origin_sampel_filename = f"{os.path.splitext(fn_save)[0]}_origin_sample.png"
    
        # 保存每个时间步的重建图像到指定的文件夹下
        input_save_path = os.path.join(recons_save_dir, input_filename)
        origin_sample_save_path = os.path.join(recons_save_dir, origin_sampel_filename)
        # visualize_components(diffusion=diffusion , noised_image=noised_list , device=device , image_path = feature_save_path)

        # diff_sample = (((origin_sample[-1] - img).clamp(-1 , 1) + 1) / 2 * 255.0).type(torch.uint8)
        x_input = ((imgs.clamp(-1 , 1) + 1) / 2 * 255.0).type(torch.uint8)
        origin_sample_res = ((origin_sample[-1].clamp(-1 , 1) + 1) / 2 * 255.0).type(torch.uint8)
        save_images(x_input , input_save_path )
        save_images(origin_sample_res , origin_sample_save_path )
        for i in range(len(de_imgs)):
            decomposed_img_path = os.path.join(recons_save_dir ,f"{os.path.splitext(fn_save)[0]}_decomposed_{i}.png")
            decomposed_image = ((de_imgs[i].clamp(-1 , 1) + 1) / 2 * 255.0).type(torch.uint8)
            diff_decompose_path = os.path.join(recons_save_dir ,f"{os.path.splitext(fn_save)[0]}_diff_{i}.png")
            diff_decompose_img = decomposed_image - origin_sample_res
            save_images(decomposed_image , decomposed_img_path )
            save_images(diff_decompose_img , diff_decompose_path )



        # 更新完成图像的数量
        have_finished_images += batch_size
        print(f"have_finished_images: {have_finished_images}")
                        
    logger.log("finish computing recons & DIRE!")


if __name__ == "__main__":
    main()
