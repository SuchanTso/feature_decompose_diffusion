MODEL_PATH="/data/usr/zsc/project/feature_decompose_diffusion/model/256x256_diffusion_uncond.pt"
SAMPLE_FLAGS="--batch_size 1 --num_samples 4  --timestep_respacing ddim30 --use_ddim True"
# 注意：这里移除了SAVE_FLAGS中的 --real_step 1，因为我们将在循环中动态设置
SAVE_FLAGS="--images_dir /data/usr/zsc/project/feature_decompose_diffusion/datasets/flowers --recons_dir /data/usr/zsc/project/feature_decompose_diffusion/imgs/edit"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
#如果--diffusion_steps为1000，--timestep_respacing为ddim20,则实际按照等间距原则在1000中选择了20个step:{0, 50, 100, ..., 950}



# 指定real_step的值从50开始到1000，刻度为50；第一个 50 是起始值，表示序列从 50 开始。第二个 50 是步长，表示每次递增的值是 50。1000 是终止值。
#real_steps=($(seq 1 1 10))
real_steps=(0)
#real_steps为0时，表示没有截断timestep步，即是完整的
#return intermediate默认是false，所以没有返回中间产物。


#diffusion_steps在/data/usr/lhr/SD_CODE/DIRE-main/guided-diffusion/guided_diffusion/script_util.py设置和上面的
# 循环遍历real_steps数组
for real_step in "${real_steps[@]}"; do
    # 动态设置real_step值，并调用python脚本
    echo "Processing with real_step = $real_step"
    python semantic_visualise.py --model_path $MODEL_PATH $MODEL_FLAGS $SAVE_FLAGS --real_step $real_step $SAMPLE_FLAGS --has_subfolder True
done
