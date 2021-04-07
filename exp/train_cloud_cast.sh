# activate your virtualenv > will need to change!


python -u run.py \
    --workspace tianyu-z \
    --epochs 200 \
    --is_training 1 \
    --dataset_name cloud_cast \
    --save_dir checkpoints/cloud_cast_predrnn \
    --gen_frm_dir results/cloud_cast_predrnn \
    --model_name predrnn_memory_decoupling \
    --reverse_input 0 \
    --img_width 128 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 36 \
    --max_iterations 80000 \
    --display_interval 1 \
    --test_interval 1 \
    --snapshot_interval 5 \
    -nbv 3
