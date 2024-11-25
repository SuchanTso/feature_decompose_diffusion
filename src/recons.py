from utils import*
from edit import*


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
        imgs = reshape_image(imgs, args.image_size)

        # noise origin image x_0 to x_T
        latent = reverse_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )
        logger.log("done noise origin image...")


        # manipulate the latent space in "ratio" part of x_T and do the rest of ddim to generate
        img = gen_featured_img(args=args,
                               model=model,
                               diffusion=diffusion,
                               batch_size=batch_size,
                               shape=(batch_size, 3, args.image_size, args.image_size),
                               clip_denoised=args.clip_denoised,
                               noise=latent,
                               model_kwargs=model_kwargs,
                               device=device,
                               ratio=0.9
                               )

            # 获取原始图像的文件夹和文件名
        original_dir_name = os.path.basename(os.path.dirname(paths[-1]))
        fn_save = os.path.basename(paths[-1])
    
        # 创建以原始图像命名的新文件夹
        recons_save_dir = os.path.join(args.recons_dir, original_dir_name, os.path.splitext(fn_save)[0])
        os.makedirs(recons_save_dir, exist_ok=True)

        input_filename = f"{os.path.splitext(fn_save)[0]}{args.real_step}_input.png"
        tem_filename = f"{os.path.splitext(fn_save)[0]}{args.real_step}_feature.png"

    
        # 保存每个时间步的重建图像到指定的文件夹下
        input_save_path = os.path.join(recons_save_dir, input_filename)
        tem_save_path = os.path.join(recons_save_dir, tem_filename)

        x_input = ((imgs.clamp(-1 , 1) + 1) / 2 * 255.0).type(torch.uint8)
        intermedient_noise = ((img.clamp(-1 , 1) + 1) / 2 * 255.0).type(torch.uint8)
        save_images(x_input , input_save_path )
        save_images(intermedient_noise , tem_save_path )


        # 更新完成图像的数量
        have_finished_images += batch_size
        print(f"have_finished_images: {have_finished_images}")
                        
    logger.log("finish computing recons & DIRE!")


if __name__ == "__main__":
    main()
