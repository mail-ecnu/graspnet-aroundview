#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dump_dir ../logs/dump_rs \
    --camera realsense \
    --method seq \
    --max_view 5 \
    --num_workers 30 \
    --dataset_root /data/Benchmark/graspnet
