from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
# from data_utils import read_data
# import tensorflow as tf
import tensorflow as tf
import tensorflow as tf_too
from tensorflow.python.training import moving_averages
# import tensorflow.contrib.slim as slim
# from vgg_preprocessing import preprocess_image
from tensorflow.python import keras
# from tensorflow.python.ops.nn import

from utils_imagenet import count_model_params, get_train_ops, strategy

from tensorflow.python.framework import graph_util

num_gpus = 2



MACC = [0, 0, 0]


def compute_conv_macc(filter_size, C_in, C_out, H_out):
    global MACC
    MACC[0] += filter_size * filter_size * C_in * C_out * H_out * H_out


def compute_sep_conv_macc(filter_size, C_in, C_out, H_out):
    global MACC
    MACC[1] += C_in * H_out * H_out * (filter_size * filter_size + C_out)


def compute_fc_macc(all_in, C_out):
    global MACC
    MACC[2] += all_in * C_out


with strategy.scope():

    def create_weight(name, shape, initializer=None, trainable=True, seed=None):
        if initializer is None:
            initializer = tf_too.keras.initializers.he_normal(seed=seed)

        w = tf.get_variable(name, shape, initializer=initializer,
                            trainable=trainable)  # ,aggregation=tf.VariableAggregation.MEAN)
        tf.summary.histogram(name=name, values=w)
        return w

with strategy.scope():

    def create_bias(name, shape, initializer=None):

        if initializer is None:
            initializer = tf.constant_initializer(0.0, dtype=tf.float32)
        return tf.get_variable(name, shape, initializer=initializer)  # ,aggregation=tf.VariableAggregation.MEAN)


    def drop_path(x, keep_prob):


        batch_size = tf.shape(x)[0]
        noise_shape = [batch_size, 1, 1, 1]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
        binary_tensor = tf.floor(random_tensor)
        x = tf.div(x, keep_prob) * binary_tensor

        return x


    def conv(x, filter_size, out_filters, stride, name="conv", padding="SAME",
             data_format="NHWC", seed=None):


        if data_format == "NHWC":
            actual_data_format = "channels_last"
        elif data_format == "NCHW":
            actual_data_format = "channels_first"
        else:
            raise NotImplementedError("Unknown data_format {}".format(data_format))

        H = x.get_shape()[1]
        H = H // stride
        In_C = x.get_shape()[-1]


        x = tf.layers.conv2d(
            x, out_filters, [filter_size, filter_size], stride, padding,
            data_format=actual_data_format,
            kernel_initializer=tf_too.keras.initializers.he_normal(seed=seed))

        return x


    def fully_connected(x, out_size, name="fc", seed=None):
        in_size = x.get_shape()[-1]
        with tf.variable_scope(name):
            w = create_weight("w", [in_size, out_size], seed=seed)
        x = tf.matmul(x, w)
        return x


    def max_pool(x, k_size, stride, padding="SAME", data_format="NHWC",
                 keep_size=False):


        if data_format == "NHWC":
            actual_data_format = "channels_last"
        elif data_format == "NCHW":
            actual_data_format = "channels_first"
        else:
            raise NotImplementedError("Unknown data_format {}".format(data_format))
        out = tf.layers.max_pooling2d(x, k_size, stride, padding,
                                      data_format=actual_data_format)

        if keep_size:
            if data_format == "NHWC":

                h_pad = (x.get_shape()[1] - out.get_shape()[1]) // 2
                w_pad = (x.get_shape()[2] - out.get_shape()[2]) // 2
                out = tf.pad(out, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]])
            elif data_format == "NCHW":
                h_pad = (x.get_shape()[2] - out.get_shape()[2]) // 2
                w_pad = (x.get_shape()[3] - out.get_shape()[3]) // 2
                out = tf.pad(out, [[0, 0], [0, 0], [h_pad, h_pad], [w_pad, w_pad]])
            else:
                raise NotImplementedError("Unknown data_format {}".format(data_format))
        return out


    def global_avg_pool(x, data_format="NHWC"):
        if data_format == "NHWC":
            x = tf.reduce_mean(x, [1, 2])
        elif data_format == "NCHW":
            x = tf.reduce_mean(x, [2, 3])
        else:
            raise NotImplementedError("Unknown data_format {}".format(data_format))
        return x


    def batch_norm(x, is_training, name="bn", decay=0.9, epsilon=1e-5,
                   data_format="NHWC"):
        if data_format == "NHWC":
            shape = [x.get_shape()[3]]
        elif data_format == "NCHW":
            shape = [x.get_shape()[1]]
        else:
            raise NotImplementedError("Unknown data_format {}".format(data_format))

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            offset = tf.get_variable(
                "offset", shape,
                initializer=tf.constant_initializer(0.0,
                                                    dtype=tf.float32))  # , aggregation=tf.VariableAggregation.MEAN)
            scale = tf.get_variable(
                "scale", shape,
                initializer=tf.constant_initializer(1.0,
                                                    dtype=tf.float32))  # , aggregation=tf.VariableAggregation.MEAN)
            moving_mean = tf.get_variable(
                "moving_mean", shape, trainable=False,
                initializer=tf.constant_initializer(0.0,
                                                    dtype=tf.float32))  # , aggregation=tf.VariableAggregation.MEAN)
            moving_variance = tf.get_variable(
                "moving_variance", shape, trainable=False,
                initializer=tf.constant_initializer(1.0,
                                                    dtype=tf.float32))  # , aggregation=tf.VariableAggregation.MEAN)

            if is_training:
                x, mean, variance = tf.nn.fused_batch_norm(
                    x, scale, offset, epsilon=epsilon, data_format=data_format,
                    is_training=True)
                update_mean = moving_averages.assign_moving_average(
                    moving_mean, mean, decay)
                update_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay)
                with tf.control_dependencies([update_mean, update_variance]):
                    x = tf.identity(x)
            else:
                x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                                 variance=moving_variance,
                                                 epsilon=epsilon, data_format=data_format,
                                                 is_training=False)
        return x


    def relu(x, leaky=0.0):
        return tf.where(tf.greater(x, 0), x, x * leaky)


    def get_ops(dataset_train, dataset_test, params):
        child_model = Model(
            dataset_train,
            dataset_test,
            num_gpus=params['num_gpus'],
            use_aux_heads=params['use_aux_heads'],
            cutout_size=params['cutout_size'],
            num_layers=params['num_layers'],
            num_cells=params['num_cells'],
            fixed_arc=params['fixed_arc'],
            out_filters_scale=params['out_filters_scale'],
            out_filters=params['out_filters'],
            keep_prob=params['keep_prob'],
            drop_path_keep_prob=params['drop_path_keep_prob'],
            num_epochs=params['num_epochs'],
            l2_reg=params['l2_reg'],
            data_format=params['data_format'],
            batch_size=params['batch_size'],
            eval_batch_size=params['eval_batch_size'],
            clip_mode="norm",
            grad_bound=params['grad_bound'],
            lr_init=params['lr'],
            lr_dec_every=params['lr_dec_every'],
            lr_dec_rate=params['lr_dec_rate'],
            lr_cosine=params['lr_cosine'],
            lr_max=params['lr_max'],
            lr_min=params['lr_min'],
            lr_T_0=params['lr_T_0'],
            lr_T_mul=params['lr_T_mul'],
            optim_algo="momentum",
            sync_replicas=params['sync_replicas'],
            num_aggregate=params['num_aggregate'],
            num_replicas=params['num_replicas'],
        )
        if params['fixed_arc'] is None:
            child_model.connect_controller(params['arch_pool'], params['arch_pool_prob'])
        else:
            child_model.connect_controller(None, None)

        with strategy.scope():

            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=keras.utils.losses_utils.ReductionV2.NONE)



            def compute_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.compat.v1.nn.compute_average_loss(per_example_loss, global_batch_size=child_model.batch_size)

            def build_train_single(data_input):
                with tf.GradientTape() as tape:
                    x_train, y_train = data_input





                    if child_model.use_aux_heads:
                        logits, aux_logits = child_model._model(x_train, is_training=True, reuse=tf.AUTO_REUSE)
                    else:
                        logits = child_model._model(x_train, is_training=True, reuse=tf.AUTO_REUSE)


                    logits = tf.nn.softmax(logits)

                    labels = tf.one_hot(y_train, depth=1000)
                    labels = 0.9 * labels + 0.1 / 1000

                    log_probs = -tf.cast(labels, dtype=tf.float32) * tf.log(
                        tf.clip_by_value(logits, 1e-10, 1.0))
                    loss = tf.reduce_sum(log_probs)
                    loss = tf.div(loss, child_model.batch_size // child_model.num_gpus)
                    tf.summary.scalar('loss', loss)


                    if child_model.use_aux_heads:

                        aux_logits = tf.nn.softmax(aux_logits)
                        log_probs = -tf.cast(labels, dtype=tf.float32) * tf.log(
                            tf.clip_by_value(aux_logits, 1e-10, 1.0))
                        aux_loss = tf.reduce_sum(log_probs)
                        aux_loss = tf.div(aux_loss, child_model.batch_size // child_model.num_gpus)
                        train_loss = loss + 0.4 * aux_loss
                    else:
                        train_loss = loss

                    train_preds = tf.argmax(logits, axis=1)
                    train_preds = tf.to_int32(train_preds)
                    train_acc = tf.equal(train_preds, y_train)
                    train_acc = tf.to_int32(train_acc)
                    train_acc = tf.reduce_sum(train_acc)
                    top_k_train_acc = tf.nn.in_top_k(logits, y_train, 5)
                    top_k_train_acc = tf.reduce_sum(tf.cast(top_k_train_acc, tf.int32))

                    tf_variables = [var for var in tf.trainable_variables() if "aux_head" not in var.name]
                    child_model.num_vars = count_model_params(tf_variables)
                    tf.logging.info("Model has {0} params".format(child_model.num_vars))

                    if child_model.l2_reg > 0:
                        l2_losses = []
                        for var in tf_variables:
                            l2_losses.append(tf.reduce_sum(var ** 2))
                        l2_loss = tf.add_n(l2_losses)
                        train_loss += child_model.l2_reg * l2_loss

                grads = tape.gradient(train_loss, tf_variables)
                grad_norm = tf.global_norm(grads)

                with tf.variable_scope(child_model.name, reuse=tf.AUTO_REUSE):

                    tf.logging.info(
                        " lr_init = {0}, lr_dec_every = {1}, num_train_batches = {2}, lr_dec_rate = {3} ".format(
                            child_model.lr_init, child_model.lr_dec_every, child_model.num_train_batches,
                            child_model.lr_dec_rate))
                    learning_rate = tf.train.exponential_decay(child_model.lr_init, child_model.global_step,
                                                               child_model.lr_dec_every,
                                                               child_model.lr_dec_rate, staircase=True)

                    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
                    opt.iterations = child_model.global_step

                    train_op = opt.apply_gradients(zip(grads, tf_variables))
                with tf.control_dependencies([train_op]):
                    train_loss = tf.identity(train_loss)
                return train_loss, train_acc, train_op, learning_rate, grad_norm, opt, top_k_train_acc

            def build_train():
                tf.logging.info("-" * 80)
                tf.logging.info("Build train graph")

                data_input = [child_model.x_train, child_model.y_train]
                per_replica_losses, per_replica_acc, child_model.train_op, lr, grad_norm, child_model.optimizer, per_replica_top_k_acc = strategy.experimental_run_v2(
                    build_train_single, args=(data_input,))

                train_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

                child_model.train_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_acc, axis=None)
                child_model.top_k_train_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_top_k_acc,
                                                              axis=None)

                child_model.lr = strategy.reduce(tf.distribute.ReduceOp.MEAN, lr, axis=None)
                child_model.grad_norm = strategy.reduce(tf.distribute.ReduceOp.MEAN, grad_norm, axis=None)


                return train_loss

            child_model.loss = build_train()

            def build_test_single(data_input):
                tf.logging.info("-" * 80)
                tf.logging.info("Build test graph")
                x_test, y_test = data_input

                logits = child_model._model(x_test, False, reuse=tf.AUTO_REUSE)
                test_preds = tf.argmax(logits, axis=1)
                test_preds = tf.to_int32(test_preds)
                test_acc = tf.equal(test_preds, y_test)
                test_acc = tf.to_int32(test_acc)
                test_acc = tf.reduce_sum(test_acc)

                top_k_test_acc = tf.nn.in_top_k(logits, y_test, 5)
                top_k_test_acc = tf.reduce_sum(tf.cast(top_k_test_acc, tf.int32))
                return test_acc, top_k_test_acc

            def build_test():
                tf.logging.info("-" * 80)
                tf.logging.info("Build train graph")

                data_input = [child_model.x_test, child_model.y_test]
                per_replica_acc, per_replica_top_k_acc = strategy.experimental_run_v2(
                    build_test_single, args=(data_input,))
                child_model.test_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_acc, axis=None)
                child_model.top_k_test_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_top_k_acc,
                                                             axis=None)

            build_test()

        tmp_ops = {
            "global_step": child_model.global_step,
            "loss": child_model.loss,
            "train_op": child_model.train_op,
            "lr": child_model.lr,
            "grad_norm": child_model.grad_norm,
            "train_acc": child_model.train_acc,
            'top_k_train_acc': child_model.top_k_train_acc,
            "optimizer": child_model.optimizer,
            "num_train_batches": child_model.num_train_batches,
            "eval_every": child_model.num_train_batches * params['eval_every_epochs'],
            "eval_func": child_model.eval_once,
            'train_init': child_model.train_iter.initializer,
            'test_init': child_model.test_iter.initializer,
            'weight': child_model.weight_w

        }
        return tmp_ops


class Model(object):
    def __init__(self,
                 dataset_train_original,
                 dataset_test_original,
                 num_gpus=2,
                 use_aux_heads=False,
                 cutout_size=None,
                 fixed_arc=None,
                 num_layers=2,
                 num_cells=5,
                 out_filters=24,
                 keep_prob=1.0,
                 drop_path_keep_prob=None,
                 batch_size=32,
                 eval_batch_size=100,
                 clip_mode=None,
                 grad_bound=None,
                 l2_reg=1e-4,
                 lr_init=0.1,
                 lr_dec_start=0,
                 lr_dec_every=10000,
                 lr_dec_rate=0.1,
                 lr_cosine=False,
                 lr_max=None,
                 lr_min=None,
                 lr_T_0=None,
                 lr_T_mul=None,
                 num_epochs=None,
                 optim_algo=None,
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 data_format="NHWC",
                 name="child",
                 seed=None,
                 baseline=0.0,
                 **kwargs
                 ):

        tf.logging.info("-" * 80)
        tf.logging.info("Build model {}".format(name))

        self.num_gpus = num_gpus

        self.cutout_size = cutout_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.clip_mode = clip_mode
        self.grad_bound = grad_bound
        self.l2_reg = l2_reg
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_rate = lr_dec_rate
        self.keep_prob = keep_prob
        self.optim_algo = optim_algo
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas
        self.data_format = data_format
        self.name = name
        self.seed = seed

        self.global_step = None
        self.valid_acc = None
        self.test_acc = None
        self.top_k_train_acc = None
        self.top_k_test_acc = None
        tf.logging.info("Build data ops")
        with tf.device("/cpu:0"):

            dataset_train_original = dataset_train_original.repeat()
            dataset_train = strategy.experimental_distribute_dataset(dataset_train_original)

            self.train_iter = dataset_train.make_initializable_iterator()

            self.x_train, self.y_train = self.train_iter.get_next()




            dataset_test_original = dataset_test_original.repeat()

            dataset_test = strategy.experimental_distribute_dataset(dataset_test_original)
            self.test_iter = dataset_test.make_initializable_iterator()

            self.x_test, self.y_test = self.test_iter.get_next()

            self.num_train_examples = 1281167
            self.num_train_batches = (
                                             self.num_train_examples + self.batch_size - 1) // self.batch_size

            self.lr_dec_every = lr_dec_every * self.num_train_batches

            self.x_valid, self.y_valid = None, None

            # valid data

            self.num_test_examples = 50000
            self.num_test_batches = (
                    (self.num_test_examples + self.eval_batch_size - 1)
                    // self.eval_batch_size)


        tf.logging.info('+' * 80)
        tf.logging.info('{}'.format(self.x_train))
        tf.logging.info('num_train_epoch = {}'.format(self.num_train_batches))
        if self.data_format == "NHWC":
            self.actual_data_format = "channels_last"
        elif self.data_format == "NCHW":
            self.actual_data_format = "channels_first"
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        self.use_aux_heads = use_aux_heads
        self.num_epochs = num_epochs
        self.num_train_steps = self.num_epochs * self.num_train_batches
        self.drop_path_keep_prob = drop_path_keep_prob
        self.lr_cosine = lr_cosine
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.out_filters = out_filters
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.fixed_arc = fixed_arc

        with strategy.scope():

            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                self.global_step = tf.train.get_or_create_global_step()

        if self.drop_path_keep_prob is not None:
            assert num_epochs is not None, "Need num_epochs to drop_path"

        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2 * pool_distance + 1]

        if self.use_aux_heads:
            self.aux_head_indices = [self.pool_layers[-1] + 1]

    def eval_once(self, sess, eval_set, feed_dict=None, verbose=False):


        assert self.global_step is not None
        global_step = sess.run(self.global_step)
        tf.logging.info("Eval at {}".format(global_step))

        if eval_set == "test":
            assert self.test_acc is not None
            num_examples = self.num_test_examples
            num_batches = self.num_test_batches
            acc_op = [self.test_acc, self.top_k_test_acc]
        else:
            raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

        total_acc = 0
        total_exp = 0
        total_top_k_acc = 0
        for batch_id in range(num_batches):
            acc, top_k_acc = sess.run(acc_op, feed_dict=feed_dict)
            total_acc += acc
            total_top_k_acc += top_k_acc
            total_exp += self.eval_batch_size
            if verbose:
                sys.stdout.write("\r{:<5d}/{:>5d}".format(total_acc, total_exp))
                sys.stdout.write("\r{:<5d}/{:>5d}".format(total_top_k_acc, total_exp))
        if verbose:
            tf.logging.info("")
        tf.logging.info("{}_accuracy: {:<6.4f}".format(
            eval_set, float(total_acc) / total_exp))
        tf.logging.info("{}_accuracy (top-5): {:<6.4f}".format(
            eval_set, float(total_top_k_acc) / total_exp))

    def _factorized_reduction(self, x, out_filters, stride, is_training):


        assert out_filters % 2 == 0, (
            "Need even number of filters when using this factorized reduction.")
        if stride == 1:
            with tf.variable_scope("path_conv"):
                inp_c = self._get_C(x)
                H = self._get_HW(x)
                w = create_weight("w", [1, 1, inp_c, out_filters])
                compute_conv_macc(1, inp_c, out_filters, H)
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                 data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)
                return x

        stride_spec = self._get_strides(stride)

        path1 = tf.nn.avg_pool(
            x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
        with tf.variable_scope("path1_conv"):
            inp_c = self._get_C(path1)
            H = self._get_HW(path1)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            compute_conv_macc(1, inp_c, out_filters // 2, H)
            path1 = tf.nn.conv2d(path1, w, [1, 1, 1, 1], "VALID",
                                 data_format=self.data_format)


        if self.data_format == "NHWC":
            pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
            path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
            concat_axis = 3
        else:
            pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
            path2 = tf.pad(x, pad_arr)[:, :, 1:, 1:]
            concat_axis = 1

        path2 = tf.nn.avg_pool(
            path2, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
        with tf.variable_scope("path2_conv"):
            inp_c = self._get_C(path2)
            H = self._get_HW(path2)
            compute_conv_macc(1, inp_c, out_filters // 2, H)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            path2 = tf.nn.conv2d(path2, w, [1, 1, 1, 1], "VALID",
                                 data_format=self.data_format)


        final_path = tf.concat(values=[path1, path2], axis=concat_axis)
        final_path = batch_norm(final_path, is_training,
                                data_format=self.data_format)

        return final_path

    def _get_C(self, x):

        if self.data_format == "NHWC":
            return x.get_shape()[-1]
        elif self.data_format == "NCHW":
            return x.get_shape()[1]
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _get_HW(self, x):

        return x.get_shape()[2]

    def _get_strides(self, stride):

        if self.data_format == "NHWC":
            return [1, stride, stride, 1]
        elif self.data_format == "NCHW":
            return [1, 1, stride, stride]
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _apply_drop_path(self, x, layer_id):
        drop_path_keep_prob = self.drop_path_keep_prob

        layer_ratio = float(layer_id + 1) / (self.num_layers + 2)
        drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)

        step_ratio = tf.to_float(self.global_step + 1) / tf.to_float(self.num_train_steps)
        step_ratio = tf.minimum(1.0, step_ratio)
        drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)

        x = drop_path(x, drop_path_keep_prob)
        return x

    def _maybe_calibrate_size(self, layers, out_filters, is_training):


        hw = [self._get_HW(layer) for layer in layers]
        c = [self._get_C(layer) for layer in layers]

        with tf.variable_scope("calibrate"):
            x = layers[0]
            if hw[0] != hw[1]:

                with tf.variable_scope("pool_x"):
                    x = tf.nn.relu(x)
                    x = self._factorized_reduction(x, out_filters, 2, is_training)
            elif c[0] != out_filters:
                with tf.variable_scope("pool_x"):
                    w = create_weight("w", [1, 1, c[0], out_filters])
                    x = tf.nn.relu(x)
                    H = self._get_HW(x)
                    compute_conv_macc(1, c[0], out_filters, H)
                    x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                     data_format=self.data_format)
                    x = batch_norm(x, is_training, data_format=self.data_format)

            y = layers[1]
            if c[1] != out_filters:
                with tf.variable_scope("pool_y"):
                    w = create_weight("w", [1, 1, c[1], out_filters])
                    y = tf.nn.relu(y)
                    H = self._get_HW(y)
                    compute_conv_macc(1, c[1], out_filters, H)
                    y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                                     data_format=self.data_format)
                    y = batch_norm(y, is_training, data_format=self.data_format)
        return [x, y]





    def _model(self, o_images, is_training, reuse=tf.AUTO_REUSE):


        global MACC
        MACC = [0, 0, 0]
        if self.fixed_arc is None:
            is_training = True

        g1 = tf.get_default_graph()
        with g1.as_default() as graph:
            with tf.variable_scope(self.name, reuse=reuse):

                images = o_images

                out_filters = self.out_filters
                with tf.variable_scope("stem_conv"):
                    w_1 = create_weight("w_1", [3, 3, 3, out_filters // 2])  # orignally, *3
                    self.weight_w = w_1
                    H = self._get_HW(images)
                    compute_conv_macc(3, 3, out_filters // 2, H // 2)
                    x = tf.nn.conv2d(
                        images, w_1, [1, 2, 2, 1], "SAME",
                        data_format=self.data_format)  # changed 2019_12_31, [1,1,1,1] SAME
                    x = batch_norm(x, is_training, data_format=self.data_format)

                with tf.variable_scope("stem_conv_1"):
                    x = tf.nn.relu(x)
                    w_2 = create_weight("w_2", [3, 3, out_filters // 2, out_filters])  # orignally, *3


                    x = tf.nn.conv2d(
                        x, w_2, [1, 2, 2, 1], "SAME",
                        data_format=self.data_format)  # changed 2019_12_31, [1,1,1,1] SAME
                    H = self._get_HW(x)
                    compute_conv_macc(3, out_filters // 2, out_filters, H)
                    x = batch_norm(x, is_training, data_format=self.data_format)
                layers = [x, x]

                with tf.variable_scope("stem_conv_2"):
                    x = tf.nn.relu(x)
                    w = create_weight("w", [3, 3, out_filters, out_filters])  # orignally, *3

                    x = tf.nn.conv2d(
                        x, w, [1, 2, 2, 1], "SAME",
                        data_format=self.data_format)  # changed 2019_12_31, [1,1,1,1] SAME

                    H = self._get_HW(x)
                    compute_conv_macc(3, out_filters, out_filters, H)
                    x = batch_norm(x, is_training, data_format=self.data_format)

                layers = [layers[-1], x]


                tf.logging.info("out_filters = {}".format(self.out_filters))

                out_filters = self.out_filters





                for layer_id in range(self.num_layers + 2):
                    with tf.variable_scope("layer_{0}".format(layer_id)):
                        if layer_id not in self.pool_layers:
                            if self.fixed_arc is None:
                                x = self._enas_layer(
                                    layer_id, layers, self.normal_arc, out_filters)
                            else:
                                x = self._fixed_layer(
                                    layer_id, layers, self.normal_arc, out_filters, 1, is_training,
                                    normal_or_reduction_cell="normal")
                        else:
                            out_filters *= 2
                            if self.fixed_arc is None:
                                x = self._factorized_reduction(x, out_filters, 2, is_training)
                                layers = [layers[-1], x]
                                x = self._enas_layer(
                                    layer_id, layers, self.reduce_arc, out_filters)
                            else:
                                x = self._fixed_layer(
                                    layer_id, layers, self.reduce_arc, out_filters, 2, is_training,
                                    normal_or_reduction_cell="reduction")
                        tf.logging.info("Layer {0:>2d}: {1}".format(layer_id, x))
                        layers = [layers[-1], x]

                    # auxiliary heads
                    self.num_aux_vars = 0
                    if (self.use_aux_heads and
                            layer_id in self.aux_head_indices
                            and is_training):
                        tf.logging.info("Using aux_head at layer {0}".format(layer_id))
                        with tf.variable_scope("aux_head"):
                            aux_logits = tf.nn.relu(x)

                            aux_logits = tf.layers.average_pooling2d(
                                aux_logits, [5, 5], [3, 3], "VALID",
                                data_format=self.actual_data_format)


                            with tf.variable_scope("proj"):
                                inp_c = self._get_C(aux_logits)
                                w = create_weight("w", [1, 1, inp_c, 128])
                                tf.logging.info("w_proj_shape {0}".format(w.shape))
                                aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                                          data_format=self.data_format)
                                aux_logits = batch_norm(aux_logits, is_training=True,
                                                        data_format=self.data_format)
                                aux_logits = tf.nn.relu(aux_logits)

                            with tf.variable_scope("avg_pool"):
                                inp_c = self._get_C(aux_logits)
                                hw = self._get_HW(aux_logits)
                                w = create_weight("w", [hw, hw, inp_c, 768])
                                tf.logging.info("w_avg_shape {0}".format(w.shape))
                                aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                                          data_format=self.data_format)
                                aux_logits = batch_norm(aux_logits, is_training=True,
                                                        data_format=self.data_format)
                                aux_logits = tf.nn.relu(aux_logits)

                            with tf.variable_scope("fc"):
                                aux_logits = global_avg_pool(aux_logits,
                                                             data_format=self.data_format)
                                inp_c = aux_logits.get_shape()[1]  # .value
                                w = create_weight("w", [inp_c, 1000])
                                tf.logging.info("w_fc_shape {0}".format(w.shape))
                                aux_logits = tf.matmul(aux_logits, w)


                        aux_head_variables = [
                            var for var in tf.trainable_variables() if (
                                    var.name.startswith(self.name) and "aux_head" in var.name)]
                        self.num_aux_vars = count_model_params(aux_head_variables)

                        tf.logging.info("Aux head uses {0} params".format(self.num_aux_vars))

                x = tf.nn.relu(x)
                x = global_avg_pool(x, data_format=self.data_format)
                if is_training and self.keep_prob is not None and self.keep_prob < 1.0:
                    x = tf.nn.dropout(x, self.keep_prob)
                with tf.variable_scope("fc"):
                    inp_c = self._get_C(x)

                    w = create_weight("w", [inp_c, 1000])
                    compute_fc_macc(inp_c, 1000)
                    x = tf.matmul(x, w, name='output')

            tf.logging.info('conv macc = {0}, sep_conv macc = {1}, fc macc = {2}'.format(MACC[0] // num_gpus,
                                                                                         MACC[1] // num_gpus,
                                                                                         MACC[2] // num_gpus))
            tf.logging.info('sum macc = {}'.format(sum(MACC) // num_gpus))

            if is_training and self.use_aux_heads:
                return x, aux_logits
            else:
                return x

    def _fixed_conv(self, x, f_size, out_filters, stride, is_training,
                    stack_convs=2):


        for conv_id in range(stack_convs):
            inp_c = self._get_C(x)
            if conv_id == 0:
                strides = self._get_strides(stride)
            else:
                strides = [1, 1, 1, 1]

            with tf.variable_scope("sep_conv_{}".format(conv_id)):
                w_depthwise = create_weight("w_depth", [f_size, f_size, inp_c, 1])
                w_pointwise = create_weight("w_point", [1, 1, inp_c, out_filters])
                x = tf.nn.relu(x)
                x = tf.nn.separable_conv2d(
                    x,
                    depthwise_filter=w_depthwise,
                    pointwise_filter=w_pointwise,
                    strides=strides, padding="SAME", data_format=self.data_format)
                H = self._get_HW(x)
                compute_sep_conv_macc(f_size, inp_c, out_filters, H)
                x = batch_norm(x, is_training, data_format=self.data_format)

        return x

    def _fixed_combine(self, layers, used, out_filters, is_training,
                       normal_or_reduction_cell="normal"):


        out_hw = min([self._get_HW(layer)
                      for i, layer in enumerate(layers) if used[i] == 0])
        out = []

        with tf.variable_scope("final_combine"):
            for i, layer in enumerate(layers):
                if used[i] == 0:
                    hw = self._get_HW(layer)
                    if hw > out_hw:
                        assert hw == out_hw * 2, ("i_hw={0} != {1}=o_hw".format(hw, out_hw))
                        with tf.variable_scope("calibrate_{0}".format(i)):
                            x = self._factorized_reduction(layer, out_filters, 2, is_training)
                    else:
                        x = layer
                    out.append(x)

            if self.data_format == "NHWC":
                out = tf.concat(out, axis=3)
            elif self.data_format == "NCHW":
                out = tf.concat(out, axis=1)
            else:
                raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        return out



    def _fixed_layer(self, layer_id, prev_layers, arc, out_filters, stride,
                     is_training, normal_or_reduction_cell="normal"):


        assert len(prev_layers) == 2
        layers = [prev_layers[0], prev_layers[1]]
        layers = self._maybe_calibrate_size(layers, out_filters,
                                            is_training=is_training)

        with tf.variable_scope("layer_base"):
            x = layers[1]
            inp_c = self._get_C(x)
            w = create_weight("w", [1, 1, inp_c, out_filters])
            x = tf.nn.relu(x)
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                             data_format=self.data_format)
            H = self._get_HW(x)

            x = batch_norm(x, is_training, data_format=self.data_format)
            layers[1] = x

        used = np.zeros([self.num_cells + 2], dtype=np.int32)
        used[0] = 1
        used[1] = 1
        f_sizes = [3, 5]
        for cell_id in range(self.num_cells):
            with tf.variable_scope("cell_{}".format(cell_id)):
                x_id = arc[4 * cell_id]
                # used[x_id] += 1
                x_op = arc[4 * cell_id + 1]
                x = layers[x_id]
                x_stride = stride if x_id in [0, 1] else 1
                with tf.variable_scope("x_conv"):
                    if x_op in [0, 1]:
                        f_size = f_sizes[x_op]
                        x = self._fixed_conv(x, f_size, out_filters, x_stride, is_training)
                    elif x_op in [2, 3]:
                        inp_c = self._get_C(x)
                        if x_op == 2:
                            x = tf.layers.average_pooling2d(
                                x, [3, 3], [x_stride, x_stride], "SAME",
                                data_format=self.actual_data_format)
                        else:
                            x = tf.layers.max_pooling2d(
                                x, [3, 3], [x_stride, x_stride], "SAME",
                                data_format=self.actual_data_format)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            x = tf.nn.relu(x)

                            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                             data_format=self.data_format)
                            H = self._get_HW(x)
                            compute_conv_macc(1, inp_c, out_filters, H)
                            x = batch_norm(x, is_training, data_format=self.data_format)
                    else:
                        inp_c = self._get_C(x)
                        if x_stride > 1:
                            assert x_stride == 2
                            x = self._factorized_reduction(x, out_filters, 2, is_training)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            x = tf.nn.relu(x)
                            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                            H = self._get_HW(x)
                            compute_conv_macc(1, inp_c, out_filters, H)
                            x = batch_norm(x, is_training, data_format=self.data_format)
                    if (x_op in [0, 1, 2, 3] and
                            self.drop_path_keep_prob is not None and
                            is_training):
                        x = self._apply_drop_path(x, layer_id)

                y_id = arc[4 * cell_id + 2]

                y_op = arc[4 * cell_id + 3]
                y = layers[y_id]
                y_stride = stride if y_id in [0, 1] else 1
                with tf.variable_scope("y_conv"):
                    if y_op in [0, 1]:
                        f_size = f_sizes[y_op]
                        y = self._fixed_conv(y, f_size, out_filters, y_stride, is_training)
                    elif y_op in [2, 3]:
                        inp_c = self._get_C(y)
                        if y_op == 2:
                            y = tf.layers.average_pooling2d(
                                y, [3, 3], [y_stride, y_stride], "SAME",
                                data_format=self.actual_data_format)
                        else:
                            y = tf.layers.max_pooling2d(
                                y, [3, 3], [y_stride, y_stride], "SAME",
                                data_format=self.actual_data_format)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            y = tf.nn.relu(y)
                            y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                                             data_format=self.data_format)
                            H = self._get_HW(y)
                            compute_conv_macc(1, inp_c, out_filters, H)
                            y = batch_norm(y, is_training, data_format=self.data_format)
                    else:
                        inp_c = self._get_C(y)
                        if y_stride > 1:
                            assert y_stride == 2
                            y = self._factorized_reduction(y, out_filters, 2, is_training)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            y = tf.nn.relu(y)
                            y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                                             data_format=self.data_format)
                            H = self._get_HW(y)
                            compute_conv_macc(1, inp_c, out_filters, H)
                            y = batch_norm(y, is_training, data_format=self.data_format)

                    if (y_op in [0, 1, 2, 3] and
                            self.drop_path_keep_prob is not None and
                            is_training):
                        y = self._apply_drop_path(y, layer_id)

                out = x + y
                layers.append(out)
        out = self._fixed_combine(layers, used, out_filters, is_training,
                                  normal_or_reduction_cell)

        return out

    def _enas_cell(self, x, curr_cell, prev_cell, op_id, out_filters):


        num_possible_inputs = curr_cell + 1

        with tf.variable_scope("avg_pool"):
            avg_pool = tf.layers.average_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
            avg_pool_c = self._get_C(avg_pool)
            if avg_pool_c != out_filters:
                with tf.variable_scope("conv"):
                    w = create_weight(
                        "w", [num_possible_inputs, avg_pool_c * out_filters])
                    w = w[prev_cell]
                    w = tf.reshape(w, [1, 1, avg_pool_c, out_filters])
                    avg_pool = tf.nn.relu(avg_pool)
                    avg_pool = tf.nn.conv2d(avg_pool, w, strides=[1, 1, 1, 1],
                                            padding="SAME", data_format=self.data_format)
                    H = self._get_HW(avg_pool)
                    compute_conv_macc(1, avg_pool_c, out_filters, H)
                    avg_pool = batch_norm(avg_pool, is_training=True,
                                          data_format=self.data_format)

        with tf.variable_scope("max_pool"):
            max_pool = tf.layers.max_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
            max_pool_c = self._get_C(max_pool)
            if max_pool_c != out_filters:
                with tf.variable_scope("conv"):
                    w = create_weight(
                        "w", [num_possible_inputs, max_pool_c * out_filters])
                    w = w[prev_cell]
                    w = tf.reshape(w, [1, 1, max_pool_c, out_filters])
                    max_pool = tf.nn.relu(max_pool)
                    max_pool = tf.nn.conv2d(max_pool, w, strides=[1, 1, 1, 1],
                                            padding="SAME", data_format=self.data_format)
                    max_pool = batch_norm(max_pool, is_training=True,
                                          data_format=self.data_format)

        x_c = self._get_C(x)
        if x_c != out_filters:
            with tf.variable_scope("x_conv"):
                w = create_weight("w", [num_possible_inputs, x_c * out_filters])
                w = w[prev_cell]
                w = tf.reshape(w, [1, 1, x_c, out_filters])
                x = tf.nn.relu(x)
                x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME",
                                 data_format=self.data_format)
                x = batch_norm(x, is_training=True, data_format=self.data_format)

        out = [
            self._enas_conv(x, curr_cell, prev_cell, 3, out_filters),
            self._enas_conv(x, curr_cell, prev_cell, 5, out_filters),
            avg_pool,
            max_pool,
            x,
        ]

        out = tf.stack(out, axis=0)
        out = out[op_id, :, :, :, :]
        return out

    def _enas_conv(self, x, curr_cell, prev_cell, filter_size, out_filters,
                   stack_conv=2):


        with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
            num_possible_inputs = curr_cell + 2
            for conv_id in range(stack_conv):
                with tf.variable_scope("stack_{0}".format(conv_id)):

                    inp_c = self._get_C(x)
                    w_depthwise = create_weight(
                        "w_depth", [num_possible_inputs, filter_size * filter_size * inp_c])
                    w_depthwise = w_depthwise[prev_cell, :]
                    w_depthwise = tf.reshape(
                        w_depthwise, [filter_size, filter_size, inp_c, 1])

                    w_pointwise = create_weight(
                        "w_point", [num_possible_inputs, inp_c * out_filters])
                    w_pointwise = w_pointwise[prev_cell, :]
                    w_pointwise = tf.reshape(w_pointwise, [1, 1, inp_c, out_filters])

                    with tf.variable_scope("bn"):
                        zero_init = tf.initializers.zeros(dtype=tf.float32)
                        one_init = tf.initializers.ones(dtype=tf.float32)
                        offset = create_weight(
                            "offset", [num_possible_inputs, out_filters],
                            initializer=zero_init)
                        scale = create_weight(
                            "scale", [num_possible_inputs, out_filters],
                            initializer=one_init)
                        offset = offset[prev_cell]
                        scale = scale[prev_cell]


                    x = tf.nn.relu(x)
                    x = tf.nn.separable_conv2d(
                        x,
                        depthwise_filter=w_depthwise,
                        pointwise_filter=w_pointwise,
                        strides=[1, 1, 1, 1], padding="SAME",
                        data_format=self.data_format)
                    x, _, _ = tf.nn.fused_batch_norm(
                        x, scale, offset, epsilon=1e-5, data_format=self.data_format,
                        is_training=True)
        return x

    def _enas_layer(self, layer_id, prev_layers, arc, out_filters):


        assert len(prev_layers) == 2, "need exactly 2 inputs"
        layers = [prev_layers[0], prev_layers[1]]
        layers = self._maybe_calibrate_size(layers, out_filters, is_training=True)

        for cell_id in range(self.num_cells):
            prev_layers = tf.stack(layers, axis=0)
            with tf.variable_scope("cell_{0}".format(cell_id)):
                with tf.variable_scope("x"):
                    x_id = arc[4 * cell_id]
                    x_op = arc[4 * cell_id + 1]
                    x = prev_layers[x_id, :, :, :, :]
                    x = self._enas_cell(x, cell_id, x_id, x_op, out_filters)


                with tf.variable_scope("y"):
                    y_id = arc[4 * cell_id + 2]
                    y_op = arc[4 * cell_id + 3]
                    y = prev_layers[y_id, :, :, :, :]
                    y = self._enas_cell(y, cell_id, y_id, y_op, out_filters)


                out = x + y

                layers.append(out)


        num_outs = self.num_cells
        out = tf.stack(layers[2:], axis=0)

        inp = prev_layers[0]
        if self.data_format == "NHWC":
            N = tf.shape(inp)[0]
            H = tf.shape(inp)[1]
            W = tf.shape(inp)[2]
            C = tf.shape(inp)[3]
            out = tf.transpose(out, [1, 2, 3, 0, 4])
            out = tf.reshape(out, [N, H, W, num_outs * out_filters])
        elif self.data_format == "NCHW":
            N = tf.shape(inp)[0]
            C = tf.shape(inp)[1]
            H = tf.shape(inp)[2]
            W = tf.shape(inp)[3]
            out = tf.transpose(out, [1, 0, 2, 3, 4])
            out = tf.reshape(out, [N, num_outs * out_filters, H, W])
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        with tf.variable_scope("final_conv"):
            w = create_weight("w", [self.num_cells, out_filters * out_filters])

            w = tf.reshape(w, [1, 1, num_outs * out_filters, out_filters])
            out = tf.nn.relu(out)
            out = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME",
                               data_format=self.data_format)
            out = batch_norm(out, is_training=True, data_format=self.data_format)

        out = tf.reshape(out, tf.shape(prev_layers[0]))

        return out

    def connect_controller(self, arch_pool=None, arch_pool_prob=None):

        assert self.fixed_arc is not None
        fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
        self.normal_arc = fixed_arc[:4 * self.num_cells]
        self.reduce_arc = fixed_arc[4 * self.num_cells:]

