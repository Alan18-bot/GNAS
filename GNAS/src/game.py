import tensorflow as tf
import numpy as np
from utils import get_train_ops
import time
import os
import math

class Game(object):
    def __init__(self,
                 model,
                 inputs_batch_size = 64,
                 num_arc_class = 10 + 1, 
                 num_players_copy = 3,
                 embedding_dim = 256,
                 lstm_hidden_size = 256,
                 lstm_num_layers = 2,
                 useAttention = True,
                 lr_init=0.01,
                 lr_dec_start=0,
                 lr_dec_every=1000,
                 lr_dec_rate=0.9,
                 l2_reg=0,
                 clip_mode=None,
                 grad_bound=None,
                 optim_algo="adam",
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 train_player_steps = 200,
                 ):
        self.train_player_steps = train_player_steps
        self.model = model
        self.inputs_batch_size = inputs_batch_size

        num_players = 0
        self.dict_player_id = {}
        for n_or_r in range(2):
            for node in range(2, self.model.num_cells + 2):
                for x_or_y in [0, 2]:
                    for prev_node in range(0, node):
                        for op in range(self.model.num_ops):
                            self.dict_player_id[num_players] = (n_or_r,node, prev_node, x_or_y, op)
                          
                            num_players += 1
        self.num_players = num_players
        self.feed_num_players_copy_list = tf.placeholder(shape=(num_players), dtype=tf.int32,
                                         name='feed_num_players_copy_list')
        self.num_arc_class = num_arc_class

        
        self.START_TOKEN = -1
        self.END_TOKEN = num_arc_class
       

        self.seq_inputs = tf.placeholder(shape=(inputs_batch_size, num_arc_class, num_players), dtype=tf.int32,
                                         name='seq_inputs')

        

        self.seq_inputs_length = tf.ones([inputs_batch_size], dtype=tf.int32) * num_arc_class

        self.reward_value_list = tf.placeholder(shape=(inputs_batch_size,num_arc_class), dtype=tf.float32,name = "reward_value_list")

        
        self.select_player_id = tf.placeholder(shape=(),dtype=tf.int32, name="select_player_id")
        with tf.variable_scope("environment_encoder", reuse=tf.AUTO_REUSE):
            encoder_embedding = tf.Variable(tf.random_uniform([num_players, embedding_dim]),
                                            dtype=tf.float32, name='encoder_embedding')
            
            encoder_inputs_embedded = tf.einsum('ibn,nd->ibd', tf.cast(self.seq_inputs,dtype=tf.float32), encoder_embedding)
            
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size) for _ in range(lstm_num_layers)])
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inputs_embedded,
                                                               sequence_length=self.seq_inputs_length,
                                                               dtype=tf.float32,
                                                               time_major=False)
            encoder_state = encoder_state[-1]
        self.player_out_list = []
        self.player_log_prob_list = []
        self.player_loss_list = []
        self.train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")
        self.run_op_list = []
        tf_variables_list = []
        for player_id in range(num_players):
            with tf.variable_scope("player_decoder_{}".format(player_id), reuse=tf.AUTO_REUSE):
                tokens_go = tf.ones([inputs_batch_size], dtype=tf.int32) * self.START_TOKEN
                decoder_embedding = tf.Variable(tf.random_uniform([num_arc_class, embedding_dim]),
                                                dtype=tf.float32, name='decoder_embedding')
                decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
                if useAttention:
                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=lstm_hidden_size,
                                                                               memory=encoder_outputs,
                                                                               memory_sequence_length=self.seq_inputs_length)
                    
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
                    decoder_initial_state = decoder_cell.zero_state(batch_size=inputs_batch_size, dtype=tf.float32)
                    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

               
                helper = tf.contrib.seq2seq.SampleEmbeddingHelper(decoder_embedding, tokens_go, self.END_TOKEN)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                          output_layer=tf.layers.Dense(num_arc_class))
                decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                                           maximum_iterations=self.feed_num_players_copy_list[player_id])

                decoder_logits = tf.layers.dense(decoder_outputs.rnn_output, num_arc_class) #shape = (batch_size, num_copy, num_arc)
                tmp_decoder_logits = tf.reshape(decoder_logits,shape=[-1,num_arc_class])

                
                selected_classes = tf.multinomial(tmp_decoder_logits,1)
                selected_classes = tf.reshape(selected_classes,shape=[self.inputs_batch_size,self.feed_num_players_copy_list[player_id]])
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_logits,
                                                                          labels=selected_classes)
                log_prob = tf.reduce_sum(log_prob, axis=-1) #shape=(batch_size,1)
                self.player_log_prob_list.append(log_prob) 
                self.player_out_list.append(selected_classes)  # shape = [batch_size, num_copy]
                reward = tf.gather(self.reward_value_list, selected_classes,
                                   batch_dims=1)  # the same shape as selected_classes
                reward = tf.reduce_mean(reward, axis=-1) #shape = (batch_size,1)
                loss = tf.reduce_mean(reward * log_prob )
                self.player_loss_list.append(loss)



                tf_variables = [var for var in tf.trainable_variables() if
                                var.name.startswith("player_decoder_{}".format(player_id))]
                num_shapes = len(tf_variables)
                
                tf_variables_list.append(tf_variables)
       
        tf_shape_variables_list = [ tf.stack([tf_variables_list[p_id][shape_id] for p_id in range(num_players)],axis=0) for shape_id in range(num_shapes) ]
        tf_encoder_variable = [var for var in tf.trainable_variables() if var.name.startswith("environment_encoder")]
        
        self.player_loss_list = tf.stack(self.player_loss_list, axis=0)

        train_op, lr, grad_norm, optimizer = get_train_ops(
            self.player_loss_list[self.select_player_id],
            tf_encoder_variable+ [tf_shape_variable[self.select_player_id] for tf_shape_variable in tf_shape_variables_list],
            self.train_step,
            clip_mode=clip_mode,
            grad_bound=grad_bound,
            l2_reg=l2_reg,
            lr_init=lr_init,
            lr_dec_start=lr_dec_start,
            lr_dec_every=lr_dec_every,
            lr_dec_rate=lr_dec_rate,
            optim_algo=optim_algo,
            sync_replicas=sync_replicas,
            num_aggregate=num_aggregate,
            num_replicas=num_replicas)
        self.run_op = [train_op, self.player_loss_list[self.select_player_id], self.train_step, lr, grad_norm]


    
    def generate_num_copy(self):
        num_players_copy_list = []
        self.player_id_list = []
        player_id = 0
        for n_or_r in range(2):
            for node in range(2, self.model.num_cells + 2):
                flag = [0,0]
                for prev_node in range(0, node):
                    for x_or_y in [0,2]:
                        for op in range(self.model.num_ops):
      
                            num_this_player_copy = self.num_arc_class // (node * self.model.num_ops) + 1
                            
                            if np.random.uniform(0,1)<=0.2 and flag[x_or_y // 2] == 0:
                                num_this_player_copy = 3
                                self.player_id_list.append(player_id)
                                flag[x_or_y // 2] = 1
                            else:
                                num_this_player_copy = 0
                            num_players_copy_list.append(num_this_player_copy)
                            player_id += 1

        return num_players_copy_list
                            
    

    #sample and compute reward_value_list
    def sample(self,sess,seq_inputs = None):
        self.num_players_copy_list = self.generate_num_copy()
        if seq_inputs is None:
            seq_inputs = np.zeros(shape=(self.inputs_batch_size, self.num_arc_class,self.num_players),dtype=np.int32) 
        sample_player_out_list = sess.run(self.player_out_list,feed_dict={self.seq_inputs:seq_inputs,self.feed_num_players_copy_list:self.num_players_copy_list})
        
        new_seq_inputs = np.zeros(shape=(self.inputs_batch_size, self.num_arc_class, self.num_players),
                             dtype=np.int32)

        
        batch_arc_list = [[ [ [-2 for _ in range(4*self.model.num_cells)], [-2 for _ in range(4*self.model.num_cells)]  ]  for _ in range(self.num_arc_class-1)] for _ in range(self.inputs_batch_size)]
        reward_value_list = [[-10 for _ in range(self.num_arc_class-1)] + [0] for _ in range(self.inputs_batch_size)]

        curr_arc_num = self.inputs_batch_size * self.num_arc_class

        for batch in range(self.inputs_batch_size):
            for arc_id in range(self.num_arc_class):
                for player_id in range(self.num_players):
                    if player_id in self.player_id_list:
                        seq_inputs[batch][arc_id][player_id] = 0


        for player_id,player_out in enumerate(sample_player_out_list):
            if player_id not in self.player_id_list:
                pass
            pos = self.dict_player_id[player_id]
            n_or_r = pos[0]
            node = pos[1]
            prev_node = pos[2]
            x_or_y = pos[3]
            op = pos[4]



            for batch in range(len(player_out)):
                for out_time in range(len(player_out[batch])):
                    arc_id = player_out[batch][out_time]
                    seq_inputs[batch][arc_id][player_id] = 1
                    if arc_id == self.num_arc_class-1:
                        continue
                    if reward_value_list[batch][arc_id] == -5:
                        continue

                    if batch_arc_list[batch][arc_id][n_or_r][4*(node - 2) + x_or_y] != -2:
                        reward_value_list[batch][arc_id] = -5                        
                        curr_arc_num -= 1                       
                        continue
                    else:
                        batch_arc_list[batch][arc_id][n_or_r][4 * (node - 2)+ x_or_y] = prev_node
                        batch_arc_list[batch][arc_id][n_or_r][4 * (node - 2) + x_or_y + 1] = op

        arc_reward_dict = {}
        arc_pool = []
        arc_pool_index = 0
        for batch in range(self.inputs_batch_size):
            for arc_id in range(self.num_arc_class-1):
                if reward_value_list[batch][arc_id] == -5:
                    continue
                arc_reward_dict[arc_pool_index] = (batch,arc_id)
                tmp_player_id = 0

                for n_or_r in range(2):
                    for node in range(2,self.model.num_cells+2):
                        for x_or_y in [0,2]:
                            if batch_arc_list[batch][arc_id][n_or_r][4*(node-2) + x_or_y] == -2:
                                tmp_count = 0
                                flag = 0
                                for prev_node in range(0, node):
                                    if flag >0:
                                        break
                                    for op in range(self.model.num_ops):
                                        if seq_inputs[batch][arc_id][tmp_player_id+tmp_count] > 0:
                                            batch_arc_list[batch][arc_id][n_or_r][4 * (node - 2) + x_or_y] = prev_node
                                            batch_arc_list[batch][arc_id][n_or_r][4 * (node - 2) + x_or_y + 1] = op
                                        tmp_count += 1

                                if flag == 0:
                                    batch_arc_list[batch][arc_id][n_or_r][4 * (node - 2) + x_or_y] = np.random.randint(0,node)
                                    batch_arc_list[batch][arc_id][n_or_r][4 * (node - 2) + x_or_y + 1] = np.random.randint(0,self.model.num_ops)

                            tmp_player_id += node * self.model.num_ops


                arc_pool.append(batch_arc_list[batch][arc_id])
                arc_pool_index += 1
        if len(arc_pool) == 0:
            tf.logging.error("arc_pool_len = 0")
            return reward_value_list, seq_inputs, [], []
        self.model.train(arc_pool, sess)


        vl_acc_list = self.model.valid(arc_pool, sess)
        mean = np.mean(vl_acc_list)
        new_vl_acc_list = [acc - mean for acc in vl_acc_list]
        for arc_pool_index, acc in enumerate(new_vl_acc_list):
            batch = arc_reward_dict[arc_pool_index][0]
            arc_id = arc_reward_dict[arc_pool_index][1]
            reward_value_list[batch][arc_id] = acc

        for batch in range(self.inputs_batch_size):
            for arc_id in range(self.num_arc_class):
                if reward_value_list[batch][arc_id] == -10:
                    tf.logging.error("reward value list is not finished!!!!!!!!!!!")

        rank_list = np.argsort(-np.array(vl_acc_list)).tolist()
        vl_acc_list = [vl_acc_list[r] for r in rank_list]
        arc_pool = [arc_pool[r] for r in rank_list]

        return reward_value_list, seq_inputs, arc_pool, vl_acc_list


    def train_player(self, sess,reward_value_list,seq_inputs=None ):
        if seq_inputs is None:
            seq_inputs = np.zeros(shape=(self.inputs_batch_size, self.num_arc_class, self.num_players), dtype=np.int32)

        start_time = time.time()
        tf.logging.info('train_player_start! : {}'.format(self.player_id_list))
        player_id_set = np.random.choice(self.num_players,size=28,replace=False)
        while True:

            player_id = np.random.choice(self.player_id_list)
            _, loss, gs, lr, gn = sess.run(self.run_op, feed_dict={self.seq_inputs: seq_inputs,self.select_player_id:player_id,
                                                                                   self.reward_value_list: reward_value_list,self.feed_num_players_copy_list:self.num_players_copy_list})
            if gs % 10 == 0:
                log_string = ""
                log_string += "ch_step={:<6d}".format(gs)
                log_string += " loss={:<8.6f}".format(loss)
                log_string += " lr={:<8.4f}".format(lr)
                log_string += " |g|={:<8.4f}".format(gn)

                log_string += " mins={:<10.2f}".format(float(time.time() - start_time) / 60)
                tf.logging.info(log_string)

            if gs > 0 and gs % self.train_player_steps == 0:
                tf.logging.info('train_player_finished!')

                return True

    def sample_and_train(self,sess):
        
        num_epochs = self.model.args.num_iterations
        
        seq_inputs = np.random.randint(0,2,size=(self.inputs_batch_size, self.num_arc_class,self.num_players))
        start_time = time.time()
        
        for epoch in range(num_epochs):
            reward_value_list, new_seq_inputs, arc_pool, vl_acc_list = self.sample(sess,seq_inputs)


            self.train_player(sess,reward_value_list,seq_inputs)


            seq_inputs = new_seq_inputs

            tf.logging.info('epoch_{0}, time={1}'.format(epoch, (time.time() - start_time) // 60))

            if len(arc_pool) == 0:
                continue

            with open(os.path.join(self.model.args.model_dir, 'arch_pool_{}'.format(epoch)), 'w') as fa_latest:
                with open(os.path.join(self.model.args.model_dir, 'arch_acc_{}'.format(epoch)), 'w') as fp_latest:
                    for arch, arch_acc in zip(arc_pool, vl_acc_list):
                        arch = ' '.join(map(str, arch[0])) + " " + ' '.join(map(str, arch[1]))
                        fa_latest.write('{}\n'.format(arch))
                        fp_latest.write('{}\n'.format(arch_acc))






























