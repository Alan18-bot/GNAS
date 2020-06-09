import tensorflow as tf
import numpy as np
from utils import get_train_ops
import time
import os
import math
import copy


class Game(object):
    def __init__(self,
                 model,
                 inputs_batch_size=16,
                 embedding_dim=128,
                 lstm_hidden_size=128,
                 lstm_num_layers=2,
                 useAttention=True,
                 lr_init=0.0035,
                 lr_dec_start=0,
                 lr_dec_every=10000000,
                 lr_dec_rate=0.9,
                 l2_reg=0,
                 clip_mode=None,
                 grad_bound=None,
                 optim_algo="adam",
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 train_player_steps=200,
                 ):
        self.train_player_steps = train_player_steps
        self.model = model
        self.inputs_batch_size = inputs_batch_size

        num_players = 0
        self.dict_player_id = {}

        self.pos_player_list = []
        for n_or_r in range(2):
            for node in range(2, self.model.num_cells + 2):
                for x_or_y in [0, 2]:
                    tmp_player_list = []
                    for prev_node in range(0, node):



                        self.dict_player_id[num_players] = (n_or_r, node, prev_node, x_or_y)
                        tmp_player_list.append(num_players)
                        num_players += 1
                    self.pos_player_list.append(tmp_player_list)
        self.num_players = num_players



        self.num_arc_class = 3+1

        self.START_TOKEN = -1
        self.END_TOKEN = self.num_arc_class

        self.seq_inputs = tf.placeholder(shape=(inputs_batch_size, self.num_players, self.num_arc_class*self.model.num_ops), dtype=tf.int32,
                                         name='seq_inputs')



        self.seq_inputs_length = tf.ones([inputs_batch_size], dtype=tf.int32) * self.num_players

        self.reward_value_list = tf.placeholder(shape=(inputs_batch_size,self.model.num_ops ,self.num_arc_class), dtype=tf.float32,
                                                name="reward_value_list")

        self.select_player_id = tf.placeholder(shape=(), dtype=tf.int32, name="select_player_id")
        with tf.variable_scope("environment_encoder", reuse=tf.AUTO_REUSE):
            encoder_embedding = tf.Variable(tf.random_uniform([self.num_arc_class*self.model.num_ops, embedding_dim]),
                                            dtype=tf.float32, name='encoder_embedding')

            encoder_inputs_embedded = tf.einsum('ibn,nd->ibd', tf.cast(self.seq_inputs, dtype=tf.float32),
                                                encoder_embedding)

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
        self.att_matrix_list = []
        for player_id in range(num_players):
            with tf.variable_scope("player_decoder_{}".format(player_id), reuse=tf.AUTO_REUSE):
                tokens_go = tf.ones([inputs_batch_size], dtype=tf.int32) * self.START_TOKEN
                decoder_embedding = tf.Variable(tf.random_uniform([self.num_arc_class, embedding_dim]),
                                                dtype=tf.float32, name='decoder_embedding')
                decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
                if useAttention:
                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=lstm_hidden_size,
                                                                               memory=encoder_outputs,
                                                                               memory_sequence_length=self.seq_inputs_length)

                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,alignment_history=True, output_attention=True)
                    decoder_initial_state = decoder_cell.zero_state(batch_size=inputs_batch_size, dtype=tf.float32)
                    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

                helper = tf.contrib.seq2seq.SampleEmbeddingHelper(decoder_embedding, tokens_go, self.END_TOKEN)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                          output_layer=tf.layers.Dense(self.num_arc_class))
                decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                                           maximum_iterations=self.model.num_ops
                                                                                                           )


                attention_matrices = decoder_state.alignment_history.stack(
                    name="train_attention_matrix.{}".format(player_id))

                att_matrix = tf.reduce_mean(attention_matrices,axis=1)

                self.att_matrix_list.append(att_matrix)


                decoder_logits = decoder_outputs.rnn_output


                tmp_decoder_logits = tf.reshape(decoder_logits, shape=[-1, self.num_arc_class])

                selected_classes = tf.multinomial(tmp_decoder_logits, 1)

                selected_classes = tf.reshape(selected_classes, shape=[self.inputs_batch_size,
                                                                       self.model.num_ops])

                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_logits,
                                                                          labels=selected_classes)

                reshape_selected_classes = tf.reshape(selected_classes, shape=[self.inputs_batch_size,
                                                                       self.model.num_ops,1])

                sum_log_prob = tf.reduce_sum(log_prob, axis=-1)  # shape=(batch_size,1)
                self.player_log_prob_list.append(sum_log_prob)
                self.player_out_list.append(selected_classes)  # shape = [batch_size, num_copy]
                reward = tf.gather_nd(self.reward_value_list, reshape_selected_classes,
                                   batch_dims=2)  # the same shape as selected_classes

                reshape_reward = tf.reshape(reward,shape=[self.inputs_batch_size,self.model.num_ops])
                loss = tf.reduce_sum(reshape_reward * log_prob,axis=-1)
                loss = tf.reduce_mean(loss)
                self.player_loss_list.append(loss)

                tf_variables = [var for var in tf.trainable_variables() if
                                var.name.startswith("player_decoder_{}".format(player_id))]
                num_shapes = len(tf_variables)

                tf_variables_list.append(tf_variables)

        tf_shape_variables_list = [tf.stack([tf_variables_list[p_id][shape_id] for p_id in range(num_players)], axis=0)
                                   for shape_id in range(num_shapes)]
        tf_encoder_variable = [var for var in tf.trainable_variables() if var.name.startswith("environment_encoder")]


        self.player_loss_list = tf.stack(self.player_loss_list, axis=0)

        train_op, lr, grad_norm, optimizer = get_train_ops(
            self.player_loss_list[self.select_player_id],
            tf_encoder_variable + [tf_shape_variable[self.select_player_id] for tf_shape_variable in
                                   tf_shape_variables_list],
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
        self.run_op = [train_op, self.player_loss_list[self.select_player_id], self.train_step, lr, grad_norm,self.att_matrix_list]

    def generate_num_copy(self):

        self.player_id_list = []
        while len(self.player_id_list) == 0:
            player_id = 0
            for n_or_r in range(2):
                for node in range(2, self.model.num_cells + 2):

                    for x_or_y in [0, 2]:

                        if np.random.uniform(0,1)<0.7:
                            player_id += node
                            continue
                        selected_player = np.random.randint(player_id,player_id+node)
                        self.player_id_list.append(selected_player)
                        player_id += node


        return self.player_id_list


    def sample(self, sess, seq_inputs=None):
        self.num_players_copy_list = self.generate_num_copy()
        if seq_inputs is None:
            seq_inputs = np.zeros(shape=(self.inputs_batch_size, self.num_players,self.num_arc_class*self.model.num_ops), dtype=np.int32)
        sample_player_out_list = sess.run(self.player_out_list, feed_dict={self.seq_inputs: seq_inputs})



        player_batch_arc_list = [[
            [
                [
                [[-2 for _ in range(4 * self.model.num_cells)], [-2 for _ in range(4 * self.model.num_cells)]]
                for _ in
             range(self.num_arc_class - 1)]
             for _ in range(self.model.num_ops)]

            for _ in range(self.inputs_batch_size)] for _ in range(self.num_players)]
        player_batch_arc_flag = [[
            [
                [
                    0
                    for _ in
                    range(self.num_arc_class - 1)]
                for _ in range(self.model.num_ops)]

            for _ in range(self.inputs_batch_size)] for _ in range(self.num_players)]

        batch_arc_pid_list = [[
            [[[] for _ in range(4 * self.model.num_cells)], [[] for _ in range(4 * self.model.num_cells)]] for _ in
             range(self.num_arc_class - 1)] for _ in range(self.inputs_batch_size)]
        final_reward_value_list = [[
                                    [
                                    [-10 for _ in range(self.num_arc_class - 1)] + [0]
                                    for _ in range(self.model.num_ops)
                                    ]
                                    for _ in range(self.inputs_batch_size)] for _ in range(self.num_players)]

        curr_arc_num = self.inputs_batch_size * self.num_arc_class



        for player_id in self.player_id_list:
            for batch in range(self.inputs_batch_size):

                seq_inputs[batch][player_id][:]=[0 for _ in range(self.num_arc_class*self.model.num_ops)]



        for player_id, player_out in enumerate(sample_player_out_list):
            if player_id not in self.player_id_list:
                continue
            pos = self.dict_player_id[player_id]
            tf.logging.info("player_id = {0}, pos = {1}".format(player_id, pos))
            n_or_r = pos[0]
            node = pos[1]
            prev_node = pos[2]
            x_or_y = pos[3]

            for current_op in range(self.model.num_ops):
                for batch in range(len(player_out)):
                    arc_id = player_out[batch][current_op]
                    seq_inputs[batch][player_id][current_op*self.num_arc_class+arc_id] = 1
                    if arc_id == self.num_arc_class - 1:
                        continue
                    batch_arc_pid_list[batch][arc_id][n_or_r][4 * (node - 2) + x_or_y].append((player_id,current_op))




        arc_reward_dict = {}
        arc_pool = []
        arc_pool_index = 0

        for player_id, player_out in enumerate(sample_player_out_list):
            if player_id not in self.player_id_list:
                continue
            for batch in range(len(player_out)):
                for current_op in range(self.model.num_ops):
                    for arc_id in range(self.num_arc_class-1):

                        if player_batch_arc_flag[player_id][batch][current_op][arc_id] > 0:
                            continue

                        sync_player_id_op_list = []

                        tmp_length = 0
                        for n_or_r in range(2):
                            for node in range(2, self.model.num_cells + 2):
                                for x_or_y in [0, 2]:

                                    pos = self.dict_player_id[player_id]

                                    n_or_r_p = pos[0]
                                    node_p = pos[1]
                                    prev_node_p = pos[2]
                                    x_or_y_p = pos[3]
                                    if [n_or_r, node, x_or_y] == [n_or_r_p, node_p, x_or_y_p]:

                                        player_batch_arc_list[player_id][batch][current_op][arc_id][n_or_r][
                                            4 * (node - 2) + x_or_y] = prev_node_p
                                        player_batch_arc_list[player_id][batch][current_op][arc_id][n_or_r][
                                            4 * (node - 2) + x_or_y + 1] = current_op
                                        tmp_length += node

                                    elif len(batch_arc_pid_list[batch][arc_id][n_or_r][4 * (node - 2) + x_or_y]) > 0:



                                        set_player_id_index = np.random.randint(len(batch_arc_pid_list[batch][arc_id][n_or_r][4 * (node - 2) + x_or_y]))
                                        set_player_id, set_player_op = batch_arc_pid_list[batch][arc_id][n_or_r][4 * (node - 2) + x_or_y][set_player_id_index]

                                        sync_player_id_op_list.append( (set_player_id,set_player_op))
                                        pos = self.dict_player_id[set_player_id]

                                        n_or_r_p = pos[0]
                                        node_p = pos[1]
                                        prev_node_p = pos[2]
                                        x_or_y_p = pos[3]

                                        assert n_or_r_p == n_or_r
                                        assert node_p == node
                                        assert x_or_y_p == x_or_y
                                        player_batch_arc_list[player_id][batch][current_op][arc_id][n_or_r][4 * (node - 2) + x_or_y] = prev_node_p
                                        player_batch_arc_list[player_id][batch][current_op][arc_id][n_or_r][4 * (node - 2) + x_or_y + 1] = set_player_op
                                        tmp_length += node
                                    else:
                                        flag = 0
                                        candidate_node = []
                                        candidate_op = []
                                        for prev_node in range(0, node):

                                            for op in range(self.model.num_ops):
                                                if seq_inputs[batch][tmp_length][op * self.num_arc_class + arc_id] > 0:
                                                    candidate_node.append(prev_node)
                                                    candidate_op.append(op)
                                                    flag = 1

                                            tmp_length += 1
                                        if flag == 1:
                                            sample_index = np.random.randint(0, len(candidate_node))
                                            prev_node = candidate_node[sample_index]
                                            op = candidate_op[sample_index]
                                            player_batch_arc_list[player_id][batch][current_op][arc_id][n_or_r][4 * (node - 2) + x_or_y] = prev_node
                                            player_batch_arc_list[player_id][batch][current_op][arc_id][n_or_r][4 * (node - 2) + x_or_y + 1] = op

                                        else:
                                            player_batch_arc_list[player_id][batch][current_op][arc_id][n_or_r][
                                                4 * (node - 2) + x_or_y] = np.random.randint(
                                                0, node)
                                            player_batch_arc_list[player_id][batch][current_op][arc_id][n_or_r][
                                                4 * (node - 2) + x_or_y + 1] = np.random.randint(0, self.model.num_ops)

                        arc_reward_dict[arc_pool_index] = [(player_id,batch,current_op, arc_id)]
                        arc_pool.append(player_batch_arc_list[player_id][batch][current_op][arc_id])
                        player_batch_arc_flag[player_id][batch][current_op][arc_id] = 1

                        for p_id,p_op in sync_player_id_op_list:
                            if player_batch_arc_flag[p_id][batch][p_op][arc_id] >0:
                                continue
                            arc_reward_dict[arc_pool_index].append( (p_id,batch,p_op,arc_id) )
                            player_batch_arc_flag[p_id][batch][p_op][arc_id] = 1
                            player_batch_arc_list[p_id][batch][p_op][arc_id] = player_batch_arc_list[player_id][batch][current_op][arc_id]



                        arc_pool_index += 1

        tf.logging.info("arc_pool_len = {}".format(len(arc_pool)))
        if len(arc_pool) == 0:
            tf.logging.error("arc_pool_len = 0")
            return final_reward_value_list, seq_inputs, [], []
        self.model.train(arc_pool, sess)

        vl_acc_list = self.model.valid(arc_pool, sess)
        mean = np.mean(vl_acc_list)
        new_vl_acc_list = [acc - mean for acc in vl_acc_list]
        for arc_pool_index, acc in enumerate(new_vl_acc_list):
            for p_id, batch, c_op, arc_id in arc_reward_dict[arc_pool_index]:
                final_reward_value_list[p_id][batch][c_op][arc_id] = acc*100

        for batch in range(self.inputs_batch_size):

                for p_id, player_out in enumerate(sample_player_out_list):
                    if p_id not in self.player_id_list:
                        continue

                    for c_op in range(self.model.num_ops):
                        for arc_id in range(self.num_arc_class-1):

                            if arc_id == self.num_arc_class - 1:
                                continue

                            if final_reward_value_list[p_id][batch][c_op][arc_id] == -10:
                                tf.logging.error("reward value list is not finished!!!!!!!!!!!")

        rank_list = np.argsort(-np.array(vl_acc_list)).tolist()
        vl_acc_list = [vl_acc_list[r] for r in rank_list]
        arc_pool = [arc_pool[r] for r in rank_list]



        return final_reward_value_list, seq_inputs, arc_pool, vl_acc_list

    def train_player(self, sess, reward_value_list, seq_inputs=None):
        if seq_inputs is None:
            seq_inputs = np.zeros(shape=(self.inputs_batch_size, self.num_players,self.num_arc_class*self.model.num_ops), dtype=np.int32)

        start_time = time.time()
        tf.logging.info('train_player_start! : {}'.format(self.player_id_list))

        while True:

            player_id = np.random.choice(self.player_id_list)
            _, loss, gs, lr, gn,att_matrix = sess.run(self.run_op,
                                           feed_dict={self.seq_inputs: seq_inputs, self.select_player_id: player_id,
                                                      self.reward_value_list: reward_value_list[player_id]})
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




                return att_matrix

    def sample_and_train(self, sess):

        num_epochs = self.model.args.num_iterations

        seq_inputs = np.random.randint(0, 2, size=(self.inputs_batch_size, self.num_players,self.num_arc_class*self.model.num_ops))
        start_time = time.time()

        current_arc_pool = []

        current_vl_acc_list = []
        player_att_list = []
        for epoch in range(num_epochs):
            reward_value_list, new_seq_inputs, arc_pool, vl_acc_list = self.sample(sess, seq_inputs)

            att_matrix_list=self.train_player(sess, reward_value_list, seq_inputs)

            seq_inputs = new_seq_inputs

            tf.logging.info('epoch_{0}, time={1}'.format(epoch, (time.time() - start_time) // 60))

            if len(arc_pool) == 0:
                continue



            current_arc_pool += arc_pool
            current_vl_acc_list += vl_acc_list

            rank_list = np.argsort(-np.array(current_vl_acc_list)).tolist()
            current_vl_acc_list = [current_vl_acc_list[r] for r in rank_list[:10]]
            current_arc_pool = [current_arc_pool[r] for r in rank_list[:10]]

            with open(os.path.join(self.model.args.model_dir, 'best_arch_pool_{}'.format(epoch)), 'w') as fa_latest:
                with open(os.path.join(self.model.args.model_dir, 'best_arch_acc_{}'.format(epoch)), 'w') as fp_latest:
                    for arch, arch_acc in zip(current_arc_pool, current_vl_acc_list):
                        arch = ' '.join(map(str, arch[0])) + " " + ' '.join(map(str, arch[1]))
                        fa_latest.write('{}\n'.format(arch))
                        fp_latest.write('{}\n'.format(arch_acc))



            if epoch == num_epochs -1 :


                    for p_id in range(self.num_players):
                        p_id_att_list = []
                        pos = self.dict_player_id[p_id]
                        att_matrix = att_matrix_list[p_id]

                        for op in range(self.model.num_ops):

                            r_list = np.argsort(-np.array(att_matrix[op])).tolist()


                            p_id_att = r_list[:10]
                            p_id_att_list.append(p_id_att)

                        player_att_list.append(p_id_att_list)


        return player_att_list



    def random_search(self,sess):
        num_epochs = self.model.args.num_iterations

        def sample_one_cell(num_cells, num_ops):
            arc = np.zeros([4 * num_cells], dtype=np.int32)
            for i in range(num_cells):
                arc[4 * i] = np.random.randint(0, i + 2)
                arc[4 * i + 1] = np.random.randint(0, num_ops)
                arc[4 * i + 2] = np.random.randint(0, i + 2)
                arc[4 * i + 3] = np.random.randint(0, num_ops)

            return arc
        def sample_one_arc(num_cells,num_ops):
            return [ sample_one_cell(num_cells,num_ops), sample_one_cell(num_cells,num_ops) ]

        start_time = time.time()

        current_arc_pool = []

        current_vl_acc_list = []
        for epoch in range(num_epochs):

            arc_pool = [ sample_one_arc(self.model.num_cells,self.model.num_ops) for _ in range(2000) ]

            self.model.train(arc_pool, sess)

            vl_acc_list = self.model.valid(arc_pool, sess)

            rank_list = np.argsort(-np.array(vl_acc_list)).tolist()
            vl_acc_list = [vl_acc_list[r] for r in rank_list]
            arc_pool = [arc_pool[r] for r in rank_list]

            tf.logging.info('epoch_{0}, time={1}'.format(epoch, (time.time() - start_time) // 60))

            if len(arc_pool) == 0:
                continue



            current_arc_pool += arc_pool
            current_vl_acc_list += vl_acc_list

            rank_list = np.argsort(-np.array(current_vl_acc_list)).tolist()
            current_vl_acc_list = [current_vl_acc_list[r] for r in rank_list[:10]]
            current_arc_pool = [current_arc_pool[r] for r in rank_list[:10]]

            with open(os.path.join(self.model.args.model_dir, 'best_arch_pool_{}'.format(epoch)), 'w') as fa_latest:
                with open(os.path.join(self.model.args.model_dir, 'best_arch_acc_{}'.format(epoch)), 'w') as fp_latest:
                    for arch, arch_acc in zip(current_arc_pool, current_vl_acc_list):
                        arch = ' '.join(map(str, arch[0])) + " " + ' '.join(map(str, arch[1]))
                        fa_latest.write('{}\n'.format(arch))
                        fp_latest.write('{}\n'.format(arch_acc))



    def random_search_att(self,sess,player_att_list=None):


        num_epochs = self.model.args.num_iterations



        def sample_one_arc(num_cells,num_ops):
            flag_list = [0 for _ in range(self.num_players)]
            arc = [np.zeros([4 * num_cells], dtype=np.int32),np.zeros([4 * num_cells], dtype=np.int32)]
            position = 0
            for n_or_r in range(2):
                for node in range(2, self.model.num_cells + 2):
                    for x_or_y in [0, 2]:

                        cand_player_list = self.pos_player_list[position]
                        max_flag = max( [flag_list[player] for player in cand_player_list ])
                        flag_cand_player_list = [ player for player in cand_player_list if flag_list[player] == max_flag ]
                        assert len(flag_cand_player_list) > 0


                        select_index = np.random.randint(0,len(flag_cand_player_list))
                        select_player = flag_cand_player_list[select_index]



                        pos = self.dict_player_id[select_player]

                        select_prev_node = pos[2]
                        arc[n_or_r][4*(node-2)+x_or_y] = select_prev_node

                        select_op = np.random.randint(0, num_ops)

                        arc[n_or_r][4 * (node - 2) + x_or_y+1] = select_op

                        for player in player_att_list[select_player][select_op]:
                            flag_list[player] += 1

                        position += 1

            return arc



        start_time = time.time()

        current_arc_pool = []

        current_vl_acc_list = []
        for epoch in range(num_epochs):

            arc_pool = [ sample_one_arc(self.model.num_cells,self.model.num_ops) for _ in range(2000) ]

            self.model.train(arc_pool, sess)

            vl_acc_list = self.model.valid(arc_pool, sess)

            rank_list = np.argsort(-np.array(vl_acc_list)).tolist()
            vl_acc_list = [vl_acc_list[r] for r in rank_list]
            arc_pool = [arc_pool[r] for r in rank_list]

            tf.logging.info('epoch_{0}, time={1}'.format(epoch, (time.time() - start_time) // 60))

            if len(arc_pool) == 0:
                continue



            current_arc_pool += arc_pool
            current_vl_acc_list += vl_acc_list

            rank_list = np.argsort(-np.array(current_vl_acc_list)).tolist()
            current_vl_acc_list = [current_vl_acc_list[r] for r in rank_list[:10]]
            current_arc_pool = [current_arc_pool[r] for r in rank_list[:10]]

            with open(os.path.join(self.model.args.rs_model_dir, 'best_arch_pool_{}'.format(epoch)), 'w') as fa_latest:
                with open(os.path.join(self.model.args.rs_model_dir, 'best_arch_acc_{}'.format(epoch)), 'w') as fp_latest:
                    for arch, arch_acc in zip(current_arc_pool, current_vl_acc_list):
                        arch = ' '.join(map(str, arch[0])) + " " + ' '.join(map(str, arch[1]))
                        fa_latest.write('{}\n'.format(arch))
                        fp_latest.write('{}\n'.format(arch_acc))


















