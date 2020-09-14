#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
export PYTHONPATH=./:$PYTHONPATH

DATA_DIR=data/cifar10
TRAIN_DIR = Train_outputs/
MODEL_DIR = Out_models/
RS_PLUS_TRAIN_DIR = RS_plus_Train_outputs/
RS_PLUS_MODEL_DIR = RS_plus_Out_models/



python src/train_search_GNAS_and_RS_plus.py \
  --data="$DATA_DIR" \
  --batch_size=144 \
  --train_valid_batch_size=100 \
  --cutout_size = 16 \
  --num_layers = 6 \
  --num_cells = 4 \
  --num_ops = 5 \
  --data_format="NHWC" \
  --num_iterations=50 \
  --num_epochs_per_iter = 3 \
  --l2_reg = 1e-4 \
  --grad_bound = 5.0 \
  --train_output_dir = "$TRAIN_DIR" \
  --model_dir = "$MODEL_DIR" \
  --rs_train_output_dir = "$RS_PLUS_TRAIN_DIR" \
  --rs_model_dir = "$RS_PLUS_MODEL_DIR" \
  --keep_prob = 0.9 \
  --drop_path_keep_prob = 0.6 \
  --lr_dec_every = 100 \
  --lr_init = 0.1 \
  --lr_dec_rate = 0.1 \
  --lr_max = 0.05 \
  --lr_min = 0.0005 \
  --lr_cosine \
  --lr_T_0 = 10 \
  --lr_T_mul = 2 \
  --out_filters = 16 \
  --optim_algo = 'momentum' \
  --use_aux_heads 2>&1