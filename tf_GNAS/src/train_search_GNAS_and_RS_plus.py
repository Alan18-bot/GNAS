# coding=utf-8
import os
import sys
import time
import glob
import numpy as np
import random
import logging
import argparse
import math
import tensorflow as tf
from model_search import *

from data_utils import read_data
from datetime import datetime
from tensorflow.python import pywrap_tensorflow
import copy
from utils import *

from game import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data/cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=144, help='batch size')
parser.add_argument('--train_valid_batch_size', type=int, default=100, help='batch size')
parser.add_argument('--cutout_size', type=int, default=16, help='cutout size')
parser.add_argument('--num_layers',type=int,default=6, help='number of normal cells')
parser.add_argument('--num_cells',type=int, default=4, help='number of nodes in each cell')
parser.add_argument('--num_ops',type=int,default=5,help='number of operations')
parser.add_argument('--data_format',type=str, default="NHWC", choices=['NHWC', 'NCHW'])
parser.add_argument('--num_iterations', type=int, default=50, help='num of iterations')
parser.add_argument('--num_epochs_per_iter', type=int, default=3, help='num of training epochs in each iteration')
parser.add_argument('--l2_reg', type=float, default=1e-4, help='weight decay')
parser.add_argument('--grad_bound', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--train_output_dir',type=str, default='./Train_outputs/')
parser.add_argument('--model_dir',type=str, default='./Out_models/')
parser.add_argument('--test_dir',type=str, default='./Test_models/')
parser.add_argument('--keep_prob', type=float, default=0.9)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.6)
parser.add_argument('--lr_dec_every', type=int, default=100)
parser.add_argument('--lr_init', type=float, default= 0.1)
parser.add_argument('--lr_dec_rate', type=float, default=0.1)
parser.add_argument('--lr_max', type=float, default=0.05)
parser.add_argument('--lr_min', type=float, default=0.0005)
parser.add_argument('--lr_T_0', type=int, default=10)
parser.add_argument('--lr_T_mul', type=int, default=2)
parser.add_argument('--out_filters', type=int, default=16)
parser.add_argument('--optim_algo', type=str, default='momentum')
parser.add_argument('--use_aux_heads', action='store_true', default=True)
parser.add_argument('--sync_replicas', action='store_true', default=False)
parser.add_argument('--lr_cosine', action='store_true', default=True)
parser.add_argument('--num_aggregate', type=int, default=None)
parser.add_argument('--num_replicas', type=int, default=None)
parser.add_argument('--rs_train_output_dir',type=str, default='./RS_plus_Train_outputs/')
parser.add_argument('--rs_model_dir',type=str, default='./RS_plus_Out_models/')
parser.add_argument('--rs_test_dir',type=str, default='./RS_plus_Test_models/')




args = parser.parse_args()





def main(unused_argv):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    if not os.path.isdir(args.train_output_dir):
        tf.logging.info("Path {} does not exist. Creating.".format(args.train_output_dir))
        os.makedirs(args.train_output_dir)

    if not os.path.isdir(args.model_dir):
        tf.logging.info("Path {} does not exist. Creating.".format(args.model_dir))
        os.makedirs(args.model_dir)
    if not os.path.isdir(args.test_dir):
        tf.logging.info("Path {} does not exist. Creating.".format(args.test_dir))
        os.makedirs(args.test_dir)


    if not os.path.isdir(args.rs_train_output_dir):
        tf.logging.info("Path {} does not exist. Creating.".format(args.rs_train_output_dir))
        os.makedirs(args.rs_train_output_dir)

    if not os.path.isdir(args.rs_model_dir):
        tf.logging.info("Path {} does not exist. Creating.".format(args.rs_model_dir))
        os.makedirs(args.rs_model_dir)
    if not os.path.isdir(args.rs_test_dir):
        tf.logging.info("Path {} does not exist. Creating.".format(args.rs_test_dir))
        os.makedirs(args.rs_test_dir)


    images, labels = read_data(args.data)

    g1 = tf.get_default_graph()

    M = Model(images,labels,args,g1)

    M.generate_ops()

    num_valid = len(images['valid'])

    num_batch = (num_valid + args.train_valid_batch_size - 1) // args.train_valid_batch_size

    num_train_examples = np.shape(images["train"])[0]
    batch_size = args.batch_size
    tf.logging.info("batch_size is {}".format(batch_size))
    num_train_batches = (num_train_examples + batch_size - 1) // batch_size


    start_time = time.time()
    with g1.as_default():

        GA = Game(M)

        saver = tf.train.Saver(max_to_keep=3)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            args.train_output_dir, save_steps=num_train_batches*args.num_epochs_per_iter, saver=saver)

        hooks = [checkpoint_saver_hook]
        if args.sync_replicas:
            sync_replicas_hook = M.optimizer.make_session_run_hook(True)
            hooks.append(sync_replicas_hook)

        tf.logging.info("-" * 80)
        tf.logging.info("Starting session")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.train.SingularMonitoredSession(
                config=config, hooks=hooks, checkpoint_dir=args.train_output_dir) as sess:

            player_att_list = GA.sample_and_train(sess)

        saver = tf.train.Saver(max_to_keep=3)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            args.rs_train_output_dir, save_steps=num_train_batches * args.num_epochs_per_iter, saver=saver)

        hooks = [checkpoint_saver_hook]
        if args.sync_replicas:
            sync_replicas_hook = M.optimizer.make_session_run_hook(True)
            hooks.append(sync_replicas_hook)

        tf.logging.info("-" * 80)
        tf.logging.info("Starting RS session")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.train.SingularMonitoredSession(
                config=config, hooks=hooks, checkpoint_dir=args.rs_train_output_dir) as sess:


            GA.random_search_att(sess,player_att_list)



if __name__ == '__main__':
    
    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)


