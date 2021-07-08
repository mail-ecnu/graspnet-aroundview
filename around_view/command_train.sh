CUDA_VISIBLE_DEVICES=4 python train.py \
    --camera realsense \
    --dump_dir ../logs/dump_rs \
    --method rl \
    --log_dir logs/log_rs \
    --max_view 5 \
    --batch_size 2 \
    --learning_rate 0.1 \
    --num_workers 0 \
    --dataset_root /data/Benchmark/graspnet

# realsense  kinect
# CUDA_VISIBLE_DEVICES=0 python train.py --camera kinect --log_dir logs/log_kn --batch_size 2 --dataset_root /data/Benchmark/graspnet

