#!/usr/bin/env bash
nvidia-smi


export PYTHONPATH=./:$PYTHONPATH
MODEL_DIR = model

DATA_DIR = ILSVRC2012

mkdir -p $MODEL_DIR




fixed_arc=""

python multi_player_test_imagenet/train_final_imagenet_multigpu_try.py \
  --data_path="$DATA_DIR" \
  --output_dir="$MODEL_DIR" \
  --num_gpus=2 \
  --child_data_format="NHWC" \
  --child_use_aux_heads \
  --child_cutout_size=112 \
  --child_batch_size=196 \
  --child_num_epochs=630 \
  --child_eval_every_epochs=1 \
  --child_fixed_arc="$fixed_arc" \
  --child_num_layers=12 \
  --child_out_filters=48 \
  --child_num_branches=5 \
  --child_num_cells=4 \
  --child_keep_prob=0.8 \
  --child_drop_path_keep_prob=0.6 \
  --child_l2_reg=3e-5 \
  --child_lr_cosine \
  --child_lr_max=1e-1 \
  --child_lr_min=1e-3 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 2>&1


