#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
export PYTHONPATH=./:$PYTHONPATH
MODEL_DIR=model/
DATA_DIR=data/cifar10

mkdir -p $MODEL_DIR



#arc_1 best test acc: 97.40
fixed_arc="1 3 1 4 0 4 0 4 0 4 1 1 0 2 3 1 1 4 0 3 1 1 1 1 0 0 0 4 2 1 1 2"

#arc_2 best test acc: 97.36
fixed_arc="1 4 1 0 0 2 0 0 0 1 2 2 2 2 2 0 0 3 1 1 0 1 1 3 3 2 0 0 1 2 4 2"

#arc_3 best test acc: 97.36
fixed_arc="1 3 1 1 0 3 0 1 0 4 3 3 0 4 0 0 0 4 0 1 0 4 1 4 3 1 1 2 1 3 4 2"

#arc_4 best test acc: 97.26
fixed_arc="1 4 1 3 0 0 0 4 0 0 0 2 2 4 4 4 1 0 0 3 1 2 0 2 2 4 0 1 1 4 2 1"

#arc_5 best test acc 97.35
fixed_arc="0 4 0 3 1 0 1 0 0 4 2 4 2 0 2 1 0 4 0 1 2 4 1 0 0 3 1 3 0 0 3 2"



python multi_player/train_final.py \
  --data_path="$DATA_DIR" \
  --output_dir="$MODEL_DIR" \
  --child_data_format="NHWC" \
  --child_cutout_size=16 \
  --child_batch_size=96 \
  --child_num_epochs=700 \
  --child_eval_every_epochs=1 \
  --child_fixed_arc="$fixed_arc" \
  --child_use_aux_heads \
  --child_num_layers=18 \
  --child_out_filters=36 \
  --child_num_branches=5 \
  --child_num_cells=4 \
  --child_keep_prob=0.8 \
  --child_drop_path_keep_prob=0.6 \
  --child_l2_reg=2e-4 \
  --child_lr_cosine \
  --child_lr_max=0.025 \
  --child_lr_min=0.0001 \
  --child_lr_T_0=700 \
  --child_lr_T_mul=2 2>&1

