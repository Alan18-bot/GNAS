import tensorflow as tf
import tensorflow.contrib.slim as slim


import time
import numpy as np
import random
import math
from utils import *

from tensorflow.python.training import moving_averages


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
  if initializer is None:
    initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
  return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

def drop_path(x, keep_prob):

  batch_size = tf.shape(x)[0]
  noise_shape = [batch_size, 1, 1, 1]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
  binary_tensor = tf.floor(random_tensor)
  x = tf.div(x, keep_prob) * binary_tensor

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
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    scale = tf.get_variable(
      "scale", shape,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))
    moving_mean = tf.get_variable(
      "moving_mean", shape, trainable=False,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    moving_variance = tf.get_variable(
      "moving_variance", shape, trainable=False,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))

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
      h_pad = (x.get_shape()[1].value - out.get_shape()[1].value) // 2
      w_pad = (x.get_shape()[2].value - out.get_shape()[2].value) // 2
      out = tf.pad(out, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]])
    elif data_format == "NCHW":
      h_pad = (x.get_shape()[2].value - out.get_shape()[2].value) // 2
      w_pad = (x.get_shape()[3].value - out.get_shape()[3].value) // 2
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


class Model(object):
    def __init__(self,
                 images,
                 labels,
                 args,
                 graph):

        tf.logging.info("-" * 80)
        tf.logging.info("Build model {}".format('Model'))
        self.args = args
        self.cutout_size = args.cutout_size


        self.clip_mode = "norm"
        self.grad_bound = args.grad_bound
        self.l2_reg = args.l2_reg
        self.lr_init = args.lr_init
        self.lr_dec_start = 0
        self.lr_dec_rate = args.lr_dec_rate
        self.keep_prob = args.keep_prob
        self.optim_algo = args.optim_algo
        self.sync_replicas = args.sync_replicas
        self.num_aggregate = args.num_aggregate
        self.num_replicas = args.num_replicas
        self.data_format = args.data_format
        self.name = 'Model'
        self.seed = None

        self.global_step = None
        self.valid_acc = None
        self.test_acc = None
        self.valid_arc_time = None
        self.test_arc_time = None

        self.eval_batch_size = args.train_valid_batch_size

        tf.logging.info("Build data ops")
        self.graph = graph
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                # training data
                self.num_train_examples = np.shape(images["train"])[0]
                
                self.batch_size = args.batch_size
                print ("batch_size is {}".format(self.batch_size))
                self.num_train_batches = (self.num_train_examples + self.batch_size - 1) // self.batch_size
                x_train, y_train = tf.train.shuffle_batch(
                    [images["train"], labels["train"]],
                    batch_size=self.batch_size,
                    capacity=50000,
                    enqueue_many=True,
                    min_after_dequeue=0,
                    num_threads=16,
                    seed=self.seed,
                    allow_smaller_final_batch=True,
                )
                self.lr_dec_every = args.lr_dec_every * self.num_train_batches

                def _pre_process(x):
                    x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
                    x = tf.random_crop(x, [32, 32, 3], seed=self.seed)
                    x = tf.image.random_flip_left_right(x, seed=self.seed)
                    if self.cutout_size is not None:
                        mask = tf.ones([self.cutout_size, self.cutout_size], dtype=tf.int32)
                        start = tf.random_uniform([2], minval=0, maxval=32, dtype=tf.int32)
                        mask = tf.pad(mask, [[self.cutout_size + start[0], 32 - start[0]],
                                             [self.cutout_size + start[1], 32 - start[1]]])
                        mask = mask[self.cutout_size: self.cutout_size + 32,
                               self.cutout_size: self.cutout_size + 32]
                        mask = tf.reshape(mask, [32, 32, 1])
                        mask = tf.tile(mask, [1, 1, 3])
                        x = tf.where(tf.equal(mask, 0), x=x, y=tf.zeros_like(x))
                    if self.data_format == "NCHW":
                        x = tf.transpose(x, [2, 0, 1])

                    return x

                self.x_train = tf.map_fn(_pre_process, x_train, back_prop=False)
                self.y_train = y_train

                # valid data
                self.x_valid, self.y_valid = None, None
                if images["valid"] is not None:

                    images["valid_original"] = np.copy(images["valid"])
                    labels["valid_original"] = np.copy(labels["valid"])
                    if self.data_format == "NCHW":
                        images["valid"] = tf.transpose(images["valid"], [0, 3, 1, 2])
                    self.num_valid_examples = np.shape(images["valid"])[0]
                    self.num_valid_batches = (
                            (self.num_valid_examples + self.eval_batch_size - 1)
                            // self.eval_batch_size)
                  
                    self.x_valid, self.y_valid = tf.train.shuffle_batch(
                        [images["valid"], labels["valid"]],
                        batch_size=self.eval_batch_size,
                        capacity=50000,
                        enqueue_many=True,
                        min_after_dequeue=0,
                        num_threads=16,
                        seed=self.seed,
                        allow_smaller_final_batch=True,
                    )
                    


                self.tmp_ap = tf.placeholder(tf.int32, name='arch_pool')
             
                self.arch_pool_dataset = tf.data.Dataset.from_tensor_slices(self.tmp_ap)
                self.arch_pool_dataset = self.arch_pool_dataset.map(lambda x: (x[0], x[1]))
                self.arch_pool_dataset = self.arch_pool_dataset

                self.arch_pool_iter = self.arch_pool_dataset.make_initializable_iterator()
                
                self.normal_arc_for_valid,self.reduce_arc_for_valid = self.arch_pool_iter.get_next()

                self.normal_arc_for_train, self.reduce_arc_for_train = self.tmp_ap[0],self.tmp_ap[1]
                # test data
                if self.data_format == "NCHW":
                    images["test"] = tf.transpose(images["test"], [0, 3, 1, 2])
                self.num_test_examples = np.shape(images["test"])[0]
                self.num_test_batches = (
                        (self.num_test_examples + self.batch_size - 1)
                        // self.batch_size)
                self.x_test, self.y_test = tf.train.batch(
                    [images["test"], labels["test"]],
                    batch_size=self.batch_size,
                    capacity=10000,
                    enqueue_many=True,
                    num_threads=1,
                    allow_smaller_final_batch=True,
                )


        
        # cache images and labels
        self.images = images
        self.labels = labels

        if self.data_format == "NHWC":
            self.actual_data_format = "channels_last"
        elif self.data_format == "NCHW":
            self.actual_data_format = "channels_first"
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))



        self.use_aux_heads = args.use_aux_heads
        self.num_iterations = args.num_iterations
        self.num_train_steps = self.num_iterations * self.num_train_batches
        self.drop_path_keep_prob = args.drop_path_keep_prob
        self.lr_cosine = args.lr_cosine
        self.lr_max = args.lr_max
        self.lr_min = args.lr_min
        self.lr_T_0 = args.lr_T_0

        self.lr_T_mul = args.lr_T_mul
        self.out_filters = args.out_filters
        self.num_layers = args.num_layers
        self.num_cells = args.num_cells
        self.num_ops = args.num_ops




        if self.drop_path_keep_prob is not None:
            assert self.num_iterations is not None, "Need num_epochs to drop_path"

        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2 * pool_distance + 1]

        if self.use_aux_heads:
            self.aux_head_indices = [self.pool_layers[-1] + 1]

        self.graph = graph
        with self.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self.global_step = tf.train.get_or_create_global_step()





    def _model_for_train(self, images, is_training, reuse=tf.AUTO_REUSE):

        with tf.variable_scope(self.name, reuse=reuse):
            # the first two inputs
            with tf.variable_scope("stem_conv"):
                w = create_weight("w", [3, 3, 3, self.out_filters * 3])
                x = tf.nn.conv2d(
                    images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)
            if self.data_format == "NHWC":
                split_axis = 3
            elif self.data_format == "NCHW":
                split_axis = 1
            else:
                raise ValueError("Unknown data_format '{0}'".format(self.data_format))
            layers = [x, x]

            # building layers in the micro space
            out_filters = self.out_filters
            for layer_id in range(self.num_layers + 2):
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    if layer_id not in self.pool_layers:
                        self.n_or_r = 'normal'
                        x = self._enas_layer(
                                layer_id, layers, self.normal_arc_for_train, out_filters,is_training)

                    else:
                        out_filters *= 2

                        x = self._factorized_reduction(x, out_filters, 2, is_training)
                        #layers = [layers[-1],x]
                        layers = [layers[0], x]
                        self.n_or_r = 'reduce'
                        x = self._enas_layer(
                                layer_id, layers, self.reduce_arc_for_train, out_filters,is_training)

                    #tf.logging.error("num_layers = {0}, pool_layers = {1}".format(self.num_layers, self.pool_layers))
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
                            aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                                      data_format=self.data_format)
                            aux_logits = batch_norm(aux_logits, is_training=is_training,
                                                    data_format=self.data_format)
                            aux_logits = tf.nn.relu(aux_logits)

                        with tf.variable_scope("avg_pool"):
                            inp_c = self._get_C(aux_logits)
                            hw = self._get_HW(aux_logits)
                            w = create_weight("w", [hw, hw, inp_c, 768])
                            aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                                      data_format=self.data_format)
                            aux_logits = batch_norm(aux_logits, is_training=is_training,
                                                    data_format=self.data_format)
                            aux_logits = tf.nn.relu(aux_logits)

                        with tf.variable_scope("fc"):
                            aux_logits = global_avg_pool(aux_logits,
                                                         data_format=self.data_format)
                            inp_c = aux_logits.get_shape()[1].value
                            w = create_weight("w", [inp_c, 10])
                            aux_logits = tf.matmul(aux_logits, w)
                            self.aux_logits = aux_logits

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
                inp_c = x.get_shape()[-1].value
                w = create_weight("w", [inp_c, 10])
                x = tf.matmul(x, w)
        
        return x


    def _model_for_valid(self, images, is_training, reuse=tf.AUTO_REUSE):

        with tf.variable_scope(self.name, reuse=reuse):
            # the first two inputs
            with tf.variable_scope("stem_conv"):
                w = create_weight("w", [3, 3, 3, self.out_filters * 3])
                x = tf.nn.conv2d(
                    images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)

            layers = [x, x]

            # building layers in the micro space
            out_filters = self.out_filters
            for layer_id in range(self.num_layers + 2):
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    if layer_id not in self.pool_layers:

                        x = self._enas_layer(
                                layer_id, layers, self.normal_arc_for_valid, out_filters,is_training) #different from cnn!!! no z

                    else:
                        out_filters *= 2

                        x = self._factorized_reduction(x, out_filters, 2, is_training)
                        #layers = [layers[-1], x]
                        layers = [layers[0], x]
                        x = self._enas_layer(
                                layer_id, layers, self.reduce_arc_for_valid, out_filters,is_training) #different from cnn!! no z

                    #tf.logging.error("num_layers = {0}, pool_layers = {1}".format(self.num_layers, self.pool_layers))
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
                            aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                                      data_format=self.data_format)
                            aux_logits = batch_norm(aux_logits, is_training=is_training,
                                                    data_format=self.data_format)
                            aux_logits = tf.nn.relu(aux_logits)

                        with tf.variable_scope("avg_pool"):
                            inp_c = self._get_C(aux_logits)
                            hw = self._get_HW(aux_logits)
                            w = create_weight("w", [hw, hw, inp_c, 768])
                            aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                                      data_format=self.data_format)
                            aux_logits = batch_norm(aux_logits, is_training=is_training,
                                                    data_format=self.data_format)
                            aux_logits = tf.nn.relu(aux_logits)

                        with tf.variable_scope("fc"):
                            aux_logits = global_avg_pool(aux_logits,
                                                         data_format=self.data_format)
                            inp_c = aux_logits.get_shape()[1].value
                            w = create_weight("w", [inp_c, 10])
                            aux_logits = tf.matmul(aux_logits, w)
                            self.aux_logits = aux_logits

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
                inp_c = x.get_shape()[-1].value
                w = create_weight("w", [inp_c, 10])
                x = tf.matmul(x, w)


        return x



    def generate_ops(self):

        with self.graph.as_default():

            with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
                self._build_train()
                self._build_valid()

    def _build_train(self):
        tf.logging.info("-" * 80)
        tf.logging.info("Build train graph")
        logits= self._model_for_train(self.x_train, is_training=True, reuse=tf.AUTO_REUSE)
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.y_train)
        self.loss = tf.reduce_mean(log_probs)

        if self.use_aux_heads:
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.aux_logits, labels=self.y_train)
            self.aux_loss = tf.reduce_mean(log_probs)
            train_loss = self.loss + 0.4 * self.aux_loss
        else:
            train_loss = self.loss

        self.train_preds = tf.argmax(logits, axis=1)
        self.train_preds = tf.to_int32(self.train_preds)
        self.train_acc = tf.equal(self.train_preds, self.y_train)
        self.train_acc = tf.to_int32(self.train_acc)
        self.train_acc = tf.reduce_sum(self.train_acc)

        tf_variables = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
        self.num_vars = count_model_params(tf_variables)
        tf.logging.info("Model has {0} params".format(self.num_vars))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
                train_loss,
                tf_variables,
                self.global_step,
                self.num_train_steps,
                clip_mode=self.clip_mode,
                grad_bound=self.grad_bound,
                l2_reg=self.l2_reg,
                lr_init=self.lr_init,
                lr_dec_start=self.lr_dec_start,
                lr_dec_every=self.lr_dec_every,
                lr_dec_rate=self.lr_dec_rate,
                lr_cosine=self.lr_cosine,
                lr_max=self.lr_max,
                lr_min=self.lr_min,
                lr_T_0=self.lr_T_0,
                lr_T_mul=self.lr_T_mul,
                num_train_batches=self.num_train_batches,
                optim_algo=self.optim_algo,
                sync_replicas=self.sync_replicas,
                num_aggregate=self.num_aggregate,
                num_replicas=self.num_replicas
            )




    def _build_valid(self):
        if self.x_valid is not None:
            tf.logging.info("-" * 80)
            tf.logging.info("Build valid graph")

            tensor_valid= self._model_for_valid(self.x_valid, False, reuse=tf.AUTO_REUSE) #different from cnn!!!!!!!!!!!! no z in enas_layer


            self.valid_preds = tf.argmax(tensor_valid, axis=1)
            self.valid_preds = tf.to_int32(self.valid_preds)
            self.infer_valid = tf.equal(self.valid_preds, self.y_valid)
            self.infer_valid = tf.to_int32(self.infer_valid)
            self.valid_acc = tf.to_float(self.infer_valid)
            self.valid_acc = tf.reduce_mean(self.valid_acc)







    def train(self,arc_pool,sess):
        num_epoch = self.args.num_epochs_per_iter
        with self.graph.as_default():
            arc_index = 0
            pool_len = len(arc_pool)
            tf.logging.info("-" * 80)
            tf.logging.info("Starting session")

            start_time = time.time()
            # sess.run(self.global_step.assign(0),feed_dict={self.tmp_ap: arch_pool})
            # sess.run([self.arch_pool_iter.initializer,self.valid_iter.initializer], feed_dict={self.tmp_ap: arch_pool,self.tmp_x_valid:self.images['new_valid'][self.valid_e],self.tmp_y_valid:self.labels['new_valid'][self.valid_e]})
            while True:
                run_ops = [
                    self.loss,
                    self.lr,
                    self.grad_norm,
                    self.train_acc,
                    self.train_op,
                    self.global_step
                ]
                loss, lr, gn, tr_acc, _, global_step = sess.run(run_ops, feed_dict={
                    self.tmp_ap: arc_pool[np.random.choice(pool_len) ]})
                arc_index += 1

                if self.args.sync_replicas:
                    actual_step = global_step * self.num_aggregate
                else:
                    actual_step = global_step
                epoch = actual_step // self.num_train_batches
                curr_time = time.time()
                if global_step % 50 == 0:
                    log_string = ""
                    log_string += "epoch={:<6d}".format(epoch)
                    log_string += "ch_step={:<6d}".format(global_step)
                    log_string += " loss={:<8.6f}".format(loss)
                    log_string += " lr={:<8.4f}".format(lr)
                    log_string += " |g|={:<8.4f}".format(gn)
                    log_string += " tr_acc={:<3d}/{:>3d}".format(tr_acc, self.batch_size)
                    log_string += " mins={:<10.2f}".format(float(curr_time - start_time) / 60)
                    tf.logging.info(log_string)
                # print actual_step, self.num_train_steps, self.num_epochs
                    #self.valid_arc(sample_one_arc_pure(self.num_cells, self.num_ops),sess)
                #break after 5 epochs
                if actual_step > 0 and actual_step % (self.num_train_batches * num_epoch) == 0 :
                    break





    def valid(self, arc_pool, sess):

        with self.graph.as_default():
            tf.logging.info("-" * 80)
            tf.logging.info("Starting valid")
            vl_acc_list = []
            all_valid_arc_index = []
            sess.run([self.arch_pool_iter.initializer], feed_dict={self.tmp_ap: arc_pool})

            for i, _ in enumerate(arc_pool):
                # print (i)
                vl_acc_list.append(sess.run(self.valid_acc, feed_dict={self.tmp_ap: arc_pool}))

            while True:
                if len(vl_acc_list) >= 10:
                    
                    Top_ten_index_list_orginal = []
                    temp_vl_acc_list = vl_acc_list.copy()
                    for _ in range(10):
                        Top_ten_index_list_orginal.append(temp_vl_acc_list.index(max(temp_vl_acc_list)))
                        temp_vl_acc_list[temp_vl_acc_list.index(max(temp_vl_acc_list))] = -10
                else:
                    Top_ten_index_list_orginal = [i for i in range(len(vl_acc_list))]

                Top_ten_index_list = []
                for index in Top_ten_index_list_orginal:
                    if index not in all_valid_arc_index:

                        all_valid_arc_index.append(index)
                        Top_ten_index_list.append(index)
                if len(Top_ten_index_list) == 0:
                    tf.logging.info("Finish valid!")
                    return vl_acc_list


                Top_ten_arc = [arc_pool[i] for i in Top_ten_index_list]

                new_vl_acc_list = []

                new_arc_pool = [arc for arc in Top_ten_arc for _ in range(self.num_valid_batches)]
                #tf.logging.info("Here1!")
                sess.run([self.arch_pool_iter.initializer], feed_dict={self.tmp_ap: new_arc_pool})
                #tf.logging.info("Here2!")
                for arc_index in range(len(Top_ten_index_list)):
                    # print (i)
                    vl_acc = 0.0

                    for _ in range(self.num_valid_batches):
                        vl_acc += sess.run(self.valid_acc, feed_dict={self.tmp_ap: new_arc_pool})
                    #tf.logging.info("Here3!")
                    new_vl_acc_list.append(vl_acc / self.num_valid_batches)
                    vl_acc_list[Top_ten_index_list[arc_index]] = new_vl_acc_list[-1]

            tf.logging.info("Finish valid!")
            return vl_acc_list



    def get_var(self,list_of_tensors, prefix_name=None):
        if prefix_name is None:
            return [var.name for var in list_of_tensors], list_of_tensors
        else:
            specific_tensor = []
            specific_tensor_name = []
            for var in list_of_tensors:
                if var.name.startswith(prefix_name):
                    specific_tensor.append(var)
                    specific_tensor_name.append(var.name)
            return specific_tensor_name, specific_tensor



    def _factorized_reduction(self, x, out_filters, stride, is_training):
        assert out_filters % 2 == 0, (
            "Need even number of filters when using this factorized reduction.")
        if stride == 1:
            with tf.variable_scope("path_conv"):
                inp_c = self._get_C(x)
                w = create_weight("w", [1, 1, inp_c, out_filters])
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                 data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)
                return x

        stride_spec = self._get_strides(stride)
        path1 = tf.nn.avg_pool(
            x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
        with tf.variable_scope("path1_conv"):
            inp_c = self._get_C(path1)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
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
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            path2 = tf.nn.conv2d(path2, w, [1, 1, 1, 1], "VALID",
                                 data_format=self.data_format)


        final_path = tf.concat(values=[path1, path2], axis=concat_axis)
        final_path = batch_norm(final_path, is_training,
                                data_format=self.data_format)

        return final_path

    def _get_C(self, x):
        if self.data_format == "NHWC":
            return x.get_shape()[3].value
        elif self.data_format == "NCHW":
            return x.get_shape()[1].value
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _get_HW(self, x):

        return x.get_shape()[2].value

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
                assert hw[0] == 2 * hw[1]
                with tf.variable_scope("pool_x"):
                    x = tf.nn.relu(x)
                    x = self._factorized_reduction(x, out_filters, 2, is_training)
            elif c[0] != out_filters:
                with tf.variable_scope("pool_x"):
                    w = create_weight("w", [1, 1, c[0], out_filters])
                    x = tf.nn.relu(x)
                    x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                     data_format=self.data_format)
                    x = batch_norm(x, is_training, data_format=self.data_format)

            y = layers[1]
            if c[1] != out_filters:
                with tf.variable_scope("pool_y"):
                    w = create_weight("w", [1, 1, c[1], out_filters])
                    y = tf.nn.relu(y)
                    y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                                     data_format=self.data_format)
                    y = batch_norm(y, is_training, data_format=self.data_format)
        return [x, y]

    def _enas_cell(self, x, curr_cell, prev_cell, op_id, out_filters,is_training):


        num_possible_inputs = curr_cell + 2


        with tf.variable_scope("avg_pool"):
            avg_pool = tf.layers.average_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
            avg_pool_c = self._get_C(avg_pool)
            if avg_pool_c != out_filters:
                with tf.variable_scope("conv"):
                    w = create_weight(
                        "w", [num_possible_inputs, avg_pool_c * out_filters])
                    # w = tf.Print(w,["all weight: ",w," shape = ",w.shape],message='Debug message:')
                    w = w[prev_cell]
                    # w = tf.Print(w,["all weight: ",w," shape = ",w.shape],message='Debug message:')
                    w = tf.reshape(w, [1, 1, avg_pool_c, out_filters])
                    avg_pool = tf.nn.relu(avg_pool)
                    avg_pool = tf.nn.conv2d(avg_pool, w, strides=[1, 1, 1, 1],
                                            padding="SAME", data_format=self.data_format)
                    avg_pool = batch_norm(avg_pool, is_training=is_training,
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
                    max_pool = batch_norm(max_pool, is_training=is_training,
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
                x = batch_norm(x, is_training=is_training, data_format=self.data_format)

        out = [
            self._enas_conv(x, curr_cell, prev_cell, 3, out_filters,is_training),
            self._enas_conv(x, curr_cell, prev_cell, 5, out_filters,is_training),
            avg_pool,
            max_pool,
            x,
            tf.zeros_like(x)
        ]

        out = tf.stack(out, axis=0)
        out = out[op_id, :, :, :, :]
        return out


    def _enas_conv(self, x, curr_cell, prev_cell, filter_size, out_filters,is_training,
                   stack_conv=2):

        with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
            num_possible_inputs = curr_cell + 2
            for conv_id in range(stack_conv):
                with tf.variable_scope("stack_{0}".format(conv_id)):

                    inp_c = self._get_C(x)
                    w_depthwise_original = create_weight(
                        "w_depth", [num_possible_inputs, filter_size * filter_size * inp_c])
                    w_depthwise = w_depthwise_original[prev_cell, :]
                    w_depthwise = tf.reshape(
                        w_depthwise, [filter_size, filter_size, inp_c, 1])

                    w_pointwise_original = create_weight(
                        "w_point", [num_possible_inputs, inp_c * out_filters])
                    w_pointwise = w_pointwise_original[prev_cell, :]
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
                        #'''
                        moving_mean = tf.get_variable(
                            "moving_mean", [num_possible_inputs,out_filters], trainable=False,
                            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                        moving_variance = tf.get_variable(
                            "moving_variance", [num_possible_inputs,out_filters], trainable=False,
                            initializer=tf.constant_initializer(1.0, dtype=tf.float32))
                        

                        moving_mean = moving_mean[prev_cell]
                        moving_variance = moving_variance[prev_cell]

                        tmp_moving_mean = tf.get_variable(
                            "tmp_moving_mean", [out_filters], trainable=False,
                            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                        tmp_moving_variance = tf.get_variable(
                            "tmp_moving_variance", [out_filters], trainable=False,
                            initializer=tf.constant_initializer(1.0, dtype=tf.float32))

                    x = tf.nn.relu(x)
                    x = tf.nn.separable_conv2d(
                        x,
                        depthwise_filter=w_depthwise,
                        pointwise_filter=w_pointwise,
                        strides=[1, 1, 1, 1], padding="SAME",
                        data_format=self.data_format)



                    if is_training:
                        x, mean, variance = tf.nn.fused_batch_norm(
                            x, scale, offset, epsilon=1e-5, data_format=self.data_format,
                            is_training=True)
                        copy_mean = tf.assign(tmp_moving_mean,moving_mean)
                        copy_variance = tf.assign(tmp_moving_variance,moving_variance)
                        with tf.control_dependencies([copy_mean,copy_variance]):
                            update_mean = moving_averages.assign_moving_average(
                            tmp_moving_mean, mean, 0.9)
                            update_variance = moving_averages.assign_moving_average(
                            tmp_moving_variance, variance, 0.9)
                        with tf.control_dependencies([update_mean, update_variance]):
                            copy_back_mean = tf.assign(moving_mean,tmp_moving_mean)
                            copy_back_variance = tf.assign(moving_variance,tmp_moving_variance)

                        with tf.control_dependencies([copy_back_mean,copy_back_variance]):
                            x = tf.identity(x)
                    else:
                        x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                                         variance=moving_variance,
                                                         epsilon=1e-5, data_format=self.data_format,
                                                         is_training=False)

        return x



    def _enas_layer(self, layer_id, prev_layers, arc, out_filters,is_training):

        assert len(prev_layers) == 2, "need exactly 2 inputs"
        layers = [prev_layers[0], prev_layers[1]]
        layers = self._maybe_calibrate_size(layers, out_filters, is_training=is_training)
        used = []
        for cell_id in range(self.num_cells):
            prev_layers = tf.stack(layers, axis=0)
            with tf.variable_scope("cell_{0}".format(cell_id)):
                with tf.variable_scope("x"):
                    self.x_or_y = 'x'
                    x_id = arc[4 * cell_id]
                    x_op = arc[4 * cell_id + 1]
                    x = prev_layers[x_id, :, :, :, :]
                    x = self._enas_cell(x, cell_id, x_id, x_op, out_filters,is_training)
                    x_used = tf.one_hot(x_id, depth=self.num_cells + 2, dtype=tf.int32)

                with tf.variable_scope("y"):
                    self.x_or_y = 'y'
                    y_id = arc[4 * cell_id + 2]
                    y_op = arc[4 * cell_id + 3]
                    y = prev_layers[y_id, :, :, :, :]
                    y = self._enas_cell(y, cell_id, y_id, y_op, out_filters,is_training)
                    y_used = tf.one_hot(y_id, depth=self.num_cells + 2, dtype=tf.int32)

                out = x + y
                used.extend([x_used, y_used])
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
            out = batch_norm(out, is_training=is_training, data_format=self.data_format)

        out = tf.reshape(out, tf.shape(prev_layers[0]))

        return out






