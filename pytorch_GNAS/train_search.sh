#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
export PYTHONPATH=./:$PYTHONPATH




python pytorch_GNAS/train_search.py \
    --num_arc_class=3 \
    --improved_random_search 2>&1

# --num_arc_class = 1 --seed = 15
# --num_arc_class = 2 --seed = 41
# --num_arc_class = 3 --seed = 49
# --num_arc_class = 4 --seed = 75
# --num_arc_class = 5 --seed = 2