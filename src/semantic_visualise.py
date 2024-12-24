from utils import*
from edit import*
import time
from image_generator import Diffusion_processor
from datetime import date, datetime

def prepare_dirs(fn_save , args , paths):

    now_datetime = datetime.now()
    formatted_date = now_datetime.strftime("%Y-%m-%d_%H%M%S")

    input_filename = f"input.png"
    origin_sampel_filename = f"origin_sample.png"
    comp_filename = f"comp.png"
    original_dir_name = os.path.basename(os.path.dirname(paths[-1]))
    recons_save_dir = os.path.join(args.recons_dir, original_dir_name, os.path.splitext(fn_save)[0]+"_"+formatted_date)
    # 保存每个时间步的重建图像到指定的文件夹下
    input_save_path = os.path.join(recons_save_dir, input_filename)
    origin_sample_save_path = os.path.join(recons_save_dir, origin_sampel_filename)
    comp_save_path = os.path.join(recons_save_dir, comp_filename)


    return input_save_path , origin_sample_save_path , comp_save_path

def main():
    args = create_argparser().parse_args()
    init_args(args=args , logger=logger)
    logger.log("computing recons & DIRE ...")
    dp = Diffusion_processor(args)
    have_finished_images = 0

    batch_size = min(args.batch_size, args.num_samples - have_finished_images)
    model_kwargs = {}
    # if args.class_cond:
        #     classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=args.device)
        #     model_kwargs["y"] = classes
    # total_data_len = len(list(args.img_data_loader))

    while have_finished_images < args.num_samples:
        # 直接使用args.batch_size，不需要基于MPI调整批次大小
        imgs , paths ,fn_save = dp.data_update(args=args , batch_size=batch_size)

        input_save_path , origin_sample_save_path , comp_save_path = prepare_dirs(fn_save , args , paths)
        

        time1 = time.time()
        latent , noised_list = dp.diffusion_forward(args , batch_size , imgs , model_kwargs)
        logger.log("done noise origin image...")

        time2 = time.time()

        origin_sample, eps_list, adjusted_eps_list , layer_output= dp.diffusion_inverse(args , batch_size , latent ,0.4, model_kwargs)
        print(f"layer output.size = {len(layer_output)}")
        check_prepare_path(comp_save_path)
        pca_layer_output(layer_output , 4 , 5 ,comp_save_path ,args.device)

        time3 = time.time()
        logger.log(f"forward time = {time2 - time1} , reverse time = {time3 - time2} , total = {time3 - time1}")

        dp.save_image(imgs , input_save_path)
        dp.save_image(origin_sample[-1] , origin_sample_save_path )

        # 更新完成图像的数量
        have_finished_images += batch_size
        print(f"have_finished_images: {have_finished_images}")
                        
    logger.log("finish computing recons & DIRE!")


if __name__ == "__main__":
    main()
