import numpy as np
# import tensorflow as tf
import tensorflow as tf
import random
# import bidict
import copy

# from train_final_imagenet_multigpu_try import strategy

B = 5

"""
<sos>     0
0         1
1         2
2         3
3         4
4         5
5         6
identity  7
sep conv  8
max pool  9
avg pool  10
3x3       11
5x5       12
"""

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():


    def get_train_ops(
            strategy,
            gradients,
            tf_variables,
            train_step,
            num_train_steps=None,
            clip_mode=None,
            grad_bound=None,
            l2_reg=1e-4,
            lr_warmup_val=None,
            lr_warmup_steps=100,
            lr_init=0.1,
            lr_dec_start=0,
            lr_dec_every=10000,
            lr_dec_rate=0.1,
            lr_dec_min=0.0001,
            lr_cosine=False,
            lr_max=None,
            lr_min=None,
            lr_T_0=None,
            lr_T_mul=None,
            num_train_batches=None,
            optim_algo=None,
            sync_replicas=False,
            num_aggregate=None,
            num_replicas=None,
            get_grad_norms=False,
            moving_average=None):


        grads = gradients
        grad_norm = tf.global_norm(grads)

        grad_norms = {}
        for v, g in zip(tf_variables, grads):
            if v is None or g is None:
                continue
            if isinstance(g, tf.IndexedSlices):
                grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g.values ** 2))
            else:
                grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g ** 2))

        if clip_mode is not None:
            assert grad_bound is not None, "Need grad_bound to clip gradients."
            if clip_mode == "global":
                grads, _ = tf.clip_by_global_norm(grads, grad_bound)
            elif clip_mode == "norm":
                clipped = []
                for g in grads:
                    if isinstance(g, tf.IndexedSlices):
                        c_g = tf.clip_by_norm(g.values, grad_bound)
                        c_g = tf.IndexedSlices(c_g, g.indices)
                    else:
                        c_g = tf.clip_by_norm(g, grad_bound)
                    clipped.append(g)
                grads = clipped
            else:
                raise NotImplementedError("Unknown clip_mode {}".format(clip_mode))

        if lr_cosine:
            assert lr_max is not None, "Need lr_max to use lr_cosine"
            assert lr_min is not None, "Need lr_min to use lr_cosine"
            assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
            assert lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"
            assert num_train_batches is not None, ("Need num_train_batches to use"
                                                   " lr_cosine")


            learning_rate = tf.train.cosine_decay_restarts(learning_rate=lr_max, global_step=train_step,
                                                           first_decay_steps=lr_T_0 * num_train_batches, t_mul=lr_T_mul,
                                                           alpha=lr_min)



        else:


            learning_rate = tf.train.exponential_decay(lr_init, train_step, lr_dec_every * num_train_batches,
                                                       lr_dec_rate, staircase=True)

        if lr_warmup_val is not None:
            learning_rate = tf.cond(tf.less(train_step, lr_warmup_steps),
                                    lambda: lr_warmup_val, lambda: learning_rate)

        if optim_algo == "momentum":

            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        elif optim_algo == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate)

        elif optim_algo == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        else:
            raise ValueError("Unknown optim_algo {}".format(optim_algo))

        if sync_replicas:
            assert num_aggregate is not None, "Need num_aggregate to sync."
            assert num_replicas is not None, "Need num_replicas to sync."

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=num_aggregate,
                total_num_replicas=num_replicas,
                use_locking=True)


        if moving_average is not None:
            opt = tf.contrib.opt.MovingAverageOptimizer(
                opt, average_decay=moving_average)

        opt.iterations = train_step
        train_op = opt.apply_gradients(zip(grads, tf_variables))

        with tf.control_dependencies([train_op]):
            learning_rate = tf.identity(learning_rate)

        if get_grad_norms:
            return train_op, learning_rate, grad_norm, opt, grad_norms
        else:
            return train_op, learning_rate, grad_norm, opt


def count_model_params(tf_variables):


  num_vars = 0
  for var in tf_variables:
    num_vars += np.prod([dim for dim in var.get_shape()])
  return num_vars






