#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
export PYTHONPATH=./:$PYTHONPATH




python pytorch_GNAS/train_final.py \
    --auxiliary \
    --arch='GNAS' \
    --gpu=0 \
    --seed=0 \
    --cutout 2>&1




