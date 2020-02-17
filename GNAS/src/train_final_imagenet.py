from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import copy
import json
import math
import time
import model_final_imagenet as mf

from imagenet_preprocessing import preprocess_image




parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])

parser.add_argument('--data_path', type=str, default='imagenet_data/ILSVRC2012')

parser.add_argument('--output_dir', type=str, default='models')

parser.add_argument('--num_gpus', type=int, default=2)

parser.add_argument('--child_batch_size', type=int, default=128)

parser.add_argument('--child_eval_batch_size', type=int, default=128)

parser.add_argument('--child_num_epochs', type=int, default=150)

parser.add_argument('--child_lr_dec_every', type=int, default=100)

parser.add_argument('--child_num_layers', type=int, default=5)

parser.add_argument('--child_num_cells', type=int, default=5)

parser.add_argument('--child_out_filters', type=int, default=20)

parser.add_argument('--child_out_filters_scale', type=int, default=1)

parser.add_argument('--child_num_branches', type=int, default=5)

parser.add_argument('--child_num_aggregate', type=int, default=None)

parser.add_argument('--child_num_replicas', type=int, default=None)

parser.add_argument('--child_lr_T_0', type=int, default=None)

parser.add_argument('--child_lr_T_mul', type=int, default=None)

parser.add_argument('--child_cutout_size', type=int, default=None)

parser.add_argument('--child_grad_bound', type=float, default=5.0)

parser.add_argument('--child_lr', type=float, default=0.1)

parser.add_argument('--child_lr_dec_rate', type=float, default=0.1)

parser.add_argument('--child_lr_max', type=float, default=None)

parser.add_argument('--child_lr_min', type=float, default=None)

parser.add_argument('--child_keep_prob', type=float, default=0.5)

parser.add_argument('--child_drop_path_keep_prob', type=float, default=1.0)

parser.add_argument('--child_l2_reg', type=float, default=1e-4)

parser.add_argument('--child_fixed_arc', type=str, default=None)

parser.add_argument('--child_use_aux_heads', action='store_true', default=False)

parser.add_argument('--child_sync_replicas', action='store_true', default=False)

parser.add_argument('--child_lr_cosine', action='store_true', default=False)

parser.add_argument('--child_eval_every_epochs', type=int, default=1)

parser.add_argument('--child_data_format', type=str, default="NHWC", choices=['NHWC', 'NCHW'])

want_size = 224

def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           drop_remainder=False,
                           tf_data_experimental_slack=False):

  if datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = (
        datasets_num_private_threads)
    dataset = dataset.with_options(options)
    tf.compat.v1.logging.info('datasets_num_private_threads: %s',
                              datasets_num_private_threads)


  options = tf.data.Options()

  options.experimental_threading.max_intra_op_parallelism = 1
  dataset = dataset.with_options(options)

  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:

    dataset = dataset.shuffle(buffer_size=shuffle_buffer)


  dataset = dataset.repeat(num_epochs)


  dataset = dataset.map(
      lambda value: parse_record_fn(value, is_training, dtype),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  if tf_data_experimental_slack:
    options = tf.data.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)

  return dataset


def data_set_process():
    DEFAULT_IMAGE_SIZE = 224
    NUM_CHANNELS = 3
    NUM_CLASSES = 1001

    NUM_IMAGES = {
        'train': 1281167,
        'validation': 50000,
    }

    _NUM_TRAIN_FILES = 1024
    _SHUFFLE_BUFFER = 10000

    DATASET_NAME = 'ImageNet'

    def get_filenames(is_training, data_dir):

        if is_training:
            return [
                os.path.join(data_dir, 'train-%05d-of-01024' % i)
                for i in range(_NUM_TRAIN_FILES)]
        else:
            return [
                os.path.join(data_dir, 'validation-%05d-of-00128' % i)
                for i in range(128)]

    def _parse_example_proto(example_serialized):

        feature_map = {
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                                   default_value=''),
            'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                       default_value=-1),
            'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                      default_value=''),
        }
        sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)

        feature_map.update(
            {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                         'image/object/bbox/ymin',
                                         'image/object/bbox/xmax',
                                         'image/object/bbox/ymax']})

        features = tf.io.parse_single_example(serialized=example_serialized,
                                              features=feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)


        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

        return features['image/encoded'], label, bbox

    def parse_record(raw_record, is_training, dtype):

        image_buffer, label, bbox = _parse_example_proto(raw_record)

        image = preprocess_image(
            image_buffer=image_buffer,
            bbox=bbox,
            output_height=DEFAULT_IMAGE_SIZE,
            output_width=DEFAULT_IMAGE_SIZE,
            num_channels=NUM_CHANNELS,
            is_training=is_training)
        image = tf.cast(image, dtype)

        return image, label

    def input_fn(is_training,
                 data_dir,
                 batch_size,
                 num_epochs=1,
                 dtype=tf.float32,
                 datasets_num_private_threads=None,
                 parse_record_fn=parse_record,
                 input_context=None,
                 drop_remainder=False,
                 tf_data_experimental_slack=False):

        filenames = get_filenames(is_training, data_dir)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        if input_context:
            tf.compat.v1.logging.info(
                'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
                    input_context.input_pipeline_id, input_context.num_input_pipelines))
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)

        if is_training:

            dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES) #test in 2020.1.22


        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return process_record_dataset(
            dataset=dataset,
            is_training=is_training,
            batch_size=batch_size,
            shuffle_buffer=_SHUFFLE_BUFFER,
            parse_record_fn=parse_record_fn,
            num_epochs=num_epochs,
            dtype=dtype,
            datasets_num_private_threads=datasets_num_private_threads,
            drop_remainder=drop_remainder,
            tf_data_experimental_slack=tf_data_experimental_slack,
        )

    train_dataset = input_fn(True,FLAGS.data_path,FLAGS.child_batch_size * FLAGS.num_gpus)
    test_data_set = input_fn(False,FLAGS.data_path,FLAGS.child_eval_batch_size)
    return train_dataset,test_data_set



def train():
        params = get_child_model_params()
        print(params['fixed_arc'])



        g = tf.Graph()
        with g.as_default():

            with tf.device('/cpu:0'):

                tf.logging.info("-" * 80)
                tf.logging.info("Starting process data")
                dataset_train, dataset_test = data_set_process()

            ops = mf.get_ops(dataset_train, dataset_test, params)

            saver = tf.train.Saver(max_to_keep=5)
            tf.logging.info("-" * 80)
            tf.logging.info("Starting saver hook")
            checkpoint_saver_hook = tf.train.CheckpointSaverHook(
                params['model_dir'], save_steps=ops["num_train_batches"], saver=saver)
            hooks = [checkpoint_saver_hook]
            if params['sync_replicas']:
                sync_replicas_hook = ops["optimizer"].make_session_run_hook(True)
                hooks.append(sync_replicas_hook)

            tf.logging.info("-" * 80)
            tf.logging.info("Starting session")
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True



            with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=config,checkpoint_dir=params['model_dir']), hooks=hooks) as sess:

                    start_time = time.time()


                    sess.run([ops['train_init'], ops['test_init']])
                    while True:
                        run_ops = [
                            ops["loss"],
                            ops["lr"],
                            ops["grad_norm"],
                            ops["train_acc"],
                            ops['top_k_train_acc']
                        ]
                        loss, lr, gn, tr_acc,top_k_train_acc = sess.run(run_ops)

                        global_step = sess.run(ops["global_step"])



                        if params['sync_replicas']:
                            actual_step = global_step * params['num_aggregate']
                        else:
                            actual_step = global_step
                        epoch = actual_step // ops["num_train_batches"]
                        curr_time = time.time()
                        if global_step % 400 == 0:
                            log_string = ""
                            log_string += "epoch={:<6d}".format(epoch)
                            log_string += "ch_step={:<6d}".format(global_step)
                            log_string += " loss={:<8.6f}".format(loss)
                            log_string += " lr={}".format(lr)
                            log_string += " |g|={:<8.4f}".format(gn)
                            log_string += " tr_acc={:<3d}/{:>3d}".format(tr_acc, params['batch_size'])
                            log_string += " top_5_tr_acc={:<3d}/{:>3d}".format(top_k_train_acc, params['batch_size'])
                            log_string += " mins={:<10.2f}".format(float(curr_time - start_time) / 60)
                            tf.logging.info(log_string)


                        if actual_step % ops["eval_every"] == 0:
                            ops["eval_func"](sess, "test")

                        if epoch >= params['num_epochs']:
                            tf.logging.info('Training finished!')
                            break




def get_child_model_params():
    params = {
        'data_dir': FLAGS.data_path,
        'model_dir': FLAGS.output_dir,
        'num_gpus': FLAGS.num_gpus,
        'batch_size': FLAGS.child_batch_size * FLAGS.num_gpus,
        'eval_batch_size': FLAGS.child_eval_batch_size,
        'num_epochs': FLAGS.child_num_epochs,
        'lr_dec_every': FLAGS.child_lr_dec_every,
        'num_layers': FLAGS.child_num_layers,
        'num_cells': FLAGS.child_num_cells,
        'out_filters': FLAGS.child_out_filters,
        'out_filters_scale': FLAGS.child_out_filters_scale,
        'num_aggregate': FLAGS.child_num_aggregate,
        'num_replicas': FLAGS.child_num_replicas,
        'lr_T_0': FLAGS.child_lr_T_0,
        'lr_T_mul': FLAGS.child_lr_T_mul,
        'cutout_size': FLAGS.child_cutout_size,
        'grad_bound': FLAGS.child_grad_bound,
        'lr_dec_rate': FLAGS.child_lr_dec_rate,
        'lr_max': FLAGS.child_lr_max,
        'lr_min': FLAGS.child_lr_min,
        'drop_path_keep_prob': FLAGS.child_drop_path_keep_prob,
        'keep_prob': FLAGS.child_keep_prob,
        'l2_reg': FLAGS.child_l2_reg,
        'fixed_arc': FLAGS.child_fixed_arc,
        'use_aux_heads': FLAGS.child_use_aux_heads,
        'sync_replicas': FLAGS.child_sync_replicas,
        'lr_cosine': FLAGS.child_lr_cosine,
        'eval_every_epochs': FLAGS.child_eval_every_epochs,
        'data_format': FLAGS.child_data_format,
        'lr': FLAGS.child_lr,
    }
    return params


def main(unused_argv):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    if not os.path.isdir(FLAGS.output_dir):
        print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    all_params = vars(FLAGS)
    with open(os.path.join(FLAGS.output_dir, 'hparams.json'), 'w') as f:
        json.dump(all_params, f)
    train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)

