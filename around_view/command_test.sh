#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dump_dir ../logs/dump_rs \
    --camera realsense \
    --method fixed \
    --max_view 2 \
    --num_workers 20 \
    --dataset_root /data/Benchmark/graspnet
