import os
import sys
import time
import math
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network


from statistics import mean
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker




from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layer,batch_size):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size,hidden_size)
        self.linear = nn.Linear(hidden_size*len(PRIMITIVES),hidden_size)

        self.num_layer=num_layer
        self.batch_size = batch_size
        self.gru = nn.GRU(hidden_size,hidden_size,num_layer)


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(self.batch_size, -1)
        output = self.linear(embedded).view(1,self.batch_size, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layer, self.batch_size, self.hidden_size).cuda() #hiddensize = (num_layers,batch_size,hidden_size)



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length,batch_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1,self.batch_size , -1)
        embedded = self.dropout(embedded)


        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.view(self.batch_size,self.max_length,-1)).view(self.batch_size,-1) #shape = batch_size,1,hidden_size


        output = torch.cat((embedded[0], attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = output.view(self.batch_size,-1)

        output = self.out(output)
        output = F.softmax(output, dim=-1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=device)


#define our own loss
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        #x is the output tensor with shape(batch_size, num_class)
        #y is the given reward tensor with shape(batch_size, num_class)

        y = torch.tensor(y)
        y = -1 * y
        z = torch.sum(x.cuda() * y.cuda(), -1)
        z = torch.mean(z,-1)
        return z




class Game:
    def __init__(self,
                 MODEL,
                 RS_MODEL,
                 args,
                 input_batch_size=10,
                 hidden_size = 128,
                 encoder_layers = 2,
                 ):

        #we search for two cells: normal cell and reduction cell
        self.half_num_players = sum(1 for i in range(MODEL.model._steps) for _ in range(2 + i))

        self.args = args


        num_arc_class = self.args.num_arc_class + 1
        self.MODEL = MODEL
        self.RS_MODEL = RS_MODEL

        self.input_batch_size = input_batch_size

        self.hidden_size = hidden_size


        self.num_players = 2 * self.half_num_players

        self.num_nodes = self.MODEL.model._steps

        self.num_ops = len(PRIMITIVES)

        self.encoder_layers = encoder_layers

        self.num_arc_class = num_arc_class

        #use to record the choices of each player
        self.curr_player_choices = torch.tensor(np.zeros([self.input_batch_size,self.num_arc_class,self.num_players,self.num_ops]) )


        # one encoder
        self.encoder = Encoder(self.num_arc_class,self.hidden_size,self.encoder_layers,self.input_batch_size)

        self.encoder = self.encoder.cuda()

        # P decoders
        self.decoder_list = [
            AttnDecoderRNN(self.hidden_size, self.num_arc_class, self.num_players, self.input_batch_size).cuda() for _ in
            range(self.num_players)]

        self.player_out_list = []

        self.criterion = My_loss()
        self.criterion = self.criterion.cuda()

        self.loss_list = []



        self.encoder_optimizer =  optim.Adam(self.encoder.parameters(),lr=args.player_learning_rate,weight_decay=args.player_weight_decay) #optim.SGD(encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer_list = [optim.Adam(decoder.parameters(), lr=args.player_learning_rate,weight_decay=args.player_weight_decay) for decoder in self.decoder_list ]


    def select_players(self):
        selected_player_list = np.random.choice(self.num_players,size=1,replace=False)

        return selected_player_list


    #inference phase
    def sample(self,selected_player_list,encoder_input = None):


        if encoder_input == None:
            encoder_input = torch.LongTensor(np.zeros([self.num_players,self.input_batch_size,self.num_ops])).cuda()
            for player_id in range(self.num_players):
                for batch_id in range(self.input_batch_size):
                    for op_id in range(self.num_ops):
                        selected_class = np.random.choice(self.num_arc_class)
                        encoder_input[player_id][batch_id][op_id]=selected_class

                        self.curr_player_choices[batch_id][selected_class][player_id][op_id] = 1

        #sample two arcs: one is an arc with player i inside, and another is an arc without player_i.
        def sample_two_arcs_for_pi(encoder_input,current_player_choices, player_id):

            selected_player_arcs_normal = []
            selected_player_arcs_reduce = []
            for op_id in range(self.num_ops):
                for batch_id in range(self.input_batch_size):
                    arc_class_id = encoder_input[player_id][batch_id][op_id]
                    if arc_class_id == self.num_arc_class-1:
                        selected_player_arcs_normal.append(None)
                        selected_player_arcs_normal.append(None)
                        selected_player_arcs_reduce.append(None)
                        selected_player_arcs_reduce.append(None)
                        continue

                    alpha_normal_1 = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))

                    alpha_reduce_1 = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))

                    alpha_normal_2 = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))

                    alpha_reduce_2 = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))

                    temp_arc_choices = current_player_choices[batch_id][arc_class_id]

                    index = 0

                    for curr_node in range(2, self.num_nodes + 2):
                        candidate_list = []
                        uncandidate_list = []

                        flag = 0

                        for prev_node in range(curr_node):
                           for tmp_op in range(self.num_ops):
                                if index == player_id and tmp_op == op_id :
                                    alpha_normal_1[index][tmp_op] = 1
                                    flag = 1
                                elif temp_arc_choices[index][tmp_op] == 1:
                                    candidate_list.append((index, tmp_op))
                                else:
                                    uncandidate_list.append((index, tmp_op))

                           index += 1
                        if flag == 0:
                            if len(candidate_list) >= 2:
                                [prev_node1, prev_node2] = np.random.choice(len(candidate_list), size=2,
                                                                              replace=False)

                                index_1, op_1 = candidate_list[prev_node1]
                                index_2, op_2 = candidate_list[prev_node2]

                            elif len(candidate_list) == 1:
                                index_1, op_1 = candidate_list[0]

                                [prev_node2] = np.random.choice(len(uncandidate_list), size=1)

                                index_2, op_2 = uncandidate_list[prev_node2]

                            else:

                                [prev_node1, prev_node2] = np.random.choice(len(uncandidate_list), size=2,
                                                                              replace=False)

                                index_1, op_1 = uncandidate_list[prev_node1]
                                index_2, op_2 = uncandidate_list[prev_node2]

                            alpha_normal_1[index_1][op_1] = 1
                            alpha_normal_1[index_2][op_2] = 1
                            alpha_normal_2[index_1][op_1] = 1
                            alpha_normal_2[index_2][op_2] = 1
                        else:
                            if len(candidate_list) >=2:
                                [prev_node1, prev_node2] = np.random.choice(len(candidate_list), size=2,
                                                                            replace=False)

                                index_1, op_1 = candidate_list[prev_node1]
                                index_2, op_2 = candidate_list[prev_node2]

                            elif len(candidate_list) == 1:
                                [prev_node1] = np.random.choice(len(candidate_list), size=1)
                                index_1, op_1 = candidate_list[prev_node1]
                                [prev_node2] = np.random.choice(len(uncandidate_list), size=1)
                                index_2, op_2 = uncandidate_list[prev_node2]

                            else:
                                [prev_node1,prev_node2] = np.random.choice(len(uncandidate_list), size=2,replace=False)
                                index_1, op_1 = uncandidate_list[prev_node1]
                                index_2, op_2 = uncandidate_list[prev_node2]

                            alpha_normal_1[index_1][op_1] = 1
                            alpha_normal_2[index_1][op_1] = 1
                            alpha_normal_2[index_2][op_2] = 1

                    selected_player_arcs_normal.append(alpha_normal_1)
                    selected_player_arcs_normal.append(alpha_normal_2)

                    # reduce
                    for curr_node in range(2, self.num_nodes + 2):
                        candidate_list = []
                        uncandidate_list = []

                        flag = 0

                        for prev_node in range(curr_node):
                            for tmp_op in range(self.num_ops):
                                if index == player_id and tmp_op == op_id:
                                    alpha_reduce_1[index-self.half_num_players][tmp_op] = 1
                                    flag = 1
                                elif temp_arc_choices[index][tmp_op] == 1:
                                    candidate_list.append((index, tmp_op))
                                else:
                                    uncandidate_list.append((index, tmp_op))

                            index += 1
                        if flag == 0:
                            if len(candidate_list) >= 2:
                                [prev_node1, prev_node2] = np.random.choice(len(candidate_list), size=2,
                                                                            replace=False)

                                index_1, op_1 = candidate_list[prev_node1]
                                index_2, op_2 = candidate_list[prev_node2]

                            elif len(candidate_list) == 1:
                                index_1, op_1 = candidate_list[0]

                                [prev_node2] = np.random.choice(len(uncandidate_list), size=1)

                                index_2, op_2 = uncandidate_list[prev_node2]

                            else:

                                [prev_node1, prev_node2] = np.random.choice(len(uncandidate_list), size=2,
                                                                            replace=False)

                                index_1, op_1 = uncandidate_list[prev_node1]
                                index_2, op_2 = uncandidate_list[prev_node2]

                            index_1 = index_1 - self.half_num_players
                            index_2 = index_2 - self.half_num_players

                            alpha_reduce_1[index_1][op_1] = 1
                            alpha_reduce_1[index_2][op_2] = 1
                            alpha_reduce_2[index_1][op_1] = 1
                            alpha_reduce_2[index_2][op_2] = 1
                        else:
                            if len(candidate_list) >= 2:
                                [prev_node1, prev_node2] = np.random.choice(len(candidate_list), size=2,
                                                                            replace=False)

                                index_1, op_1 = candidate_list[prev_node1]
                                index_2, op_2 = candidate_list[prev_node2]

                            elif len(candidate_list) == 1:
                                [prev_node1] = np.random.choice(len(candidate_list), size=1)
                                index_1, op_1 = candidate_list[prev_node1]
                                [prev_node2] = np.random.choice(len(uncandidate_list), size=1)
                                index_2, op_2 = uncandidate_list[prev_node2]

                            else:
                                [prev_node1, prev_node2] = np.random.choice(len(uncandidate_list), size=2,
                                                                            replace=False)
                                index_1, op_1 = uncandidate_list[prev_node1]
                                index_2, op_2 = uncandidate_list[prev_node2]
                            index_1 = index_1 - self.half_num_players
                            index_2 = index_2 - self.half_num_players

                            alpha_reduce_1[index_1][op_1] = 1
                            alpha_reduce_2[index_1][op_1] = 1
                            alpha_reduce_2[index_2][op_2] = 1

                    selected_player_arcs_reduce.append(alpha_reduce_1)
                    selected_player_arcs_reduce.append(alpha_reduce_2)



            return selected_player_arcs_normal,selected_player_arcs_reduce


        #sample an arc for each group
        def sample_arcs(current_player_choices):
            arcs_normal = []
            arcs_reduce = []

            for batch_id in range(self.input_batch_size):
                for arc_class_id in range(self.num_arc_class-1):

                    alpha_normal = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))

                    alpha_reduce = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))





                    temp_arc_choices = current_player_choices[batch_id][arc_class_id]

                    index = 0

                    for curr_node in range(2, self.num_nodes + 2):
                        candidate_list = []
                        uncandidate_list = []

                        #flag = 0

                        for prev_node in range(curr_node):
                           for ops in range(self.num_ops):

                                if temp_arc_choices[index][ops] == 1:
                                    candidate_list.append((index, ops))
                                else:
                                    uncandidate_list.append((index, ops))

                           index += 1

                        if len(candidate_list) >= 2:
                            [prev_node1, prev_node2] = np.random.choice(len(candidate_list), size=2,
                                                                              replace=False)

                            index_1, op_1 = candidate_list[prev_node1]
                            index_2, op_2 = candidate_list[prev_node2]

                        elif len(candidate_list) == 1:
                            index_1, op_1 = candidate_list[0]

                            [prev_node2] = np.random.choice(len(uncandidate_list), size=1)

                            index_2, op_2 = uncandidate_list[prev_node2]

                        else:

                            [prev_node1, prev_node2] = np.random.choice(len(uncandidate_list), size=2,
                                                                              replace=False)

                            index_1, op_1 = uncandidate_list[prev_node1]
                            index_2, op_2 = uncandidate_list[prev_node2]

                        alpha_normal[index_1][op_1] = 1
                        alpha_normal[index_2][op_2] = 1


                    arcs_normal.append(alpha_normal)


                    # reduce
                    for curr_node in range(2, self.num_nodes + 2):
                        candidate_list = []
                        uncandidate_list = []


                        for prev_node in range(curr_node):
                            for ops in range(self.num_ops):

                                if temp_arc_choices[index][ops] == 1:
                                    candidate_list.append((index, ops))
                                else:
                                    uncandidate_list.append((index, ops))

                            index += 1

                        if len(candidate_list) >= 2:
                            [prev_node1, prev_node2] = np.random.choice(len(candidate_list), size=2,
                                                                            replace=False)

                            index_1, op_1 = candidate_list[prev_node1]
                            index_2, op_2 = candidate_list[prev_node2]

                        elif len(candidate_list) == 1:
                            index_1, op_1 = candidate_list[0]

                            [prev_node2] = np.random.choice(len(uncandidate_list), size=1)

                            index_2, op_2 = uncandidate_list[prev_node2]

                        else:

                            [prev_node1, prev_node2] = np.random.choice(len(uncandidate_list), size=2,
                                                                            replace=False)

                            index_1, op_1 = uncandidate_list[prev_node1]
                            index_2, op_2 = uncandidate_list[prev_node2]

                        index_1 = index_1 - self.half_num_players
                        index_2 = index_2 - self.half_num_players

                        alpha_reduce[index_1][op_1] = 1
                        alpha_reduce[index_2][op_2] = 1



                    arcs_reduce.append(alpha_reduce)




            return arcs_normal,arcs_reduce


        self.old_normal_arcs = []
        self.old_reduce_arcs = []
        for p_id in selected_player_list:
            p_id_arcs_normal,p_id_arcs_reduce = sample_two_arcs_for_pi(encoder_input,self.curr_player_choices,p_id)

            self.old_normal_arcs = self.old_normal_arcs + p_id_arcs_normal
            self.old_reduce_arcs = self.old_reduce_arcs + p_id_arcs_reduce

        arcs_normal,arcs_reduce = sample_arcs(self.curr_player_choices)
        self.old_normal_arcs = self.old_normal_arcs + arcs_normal
        self.old_reduce_arcs = self.old_reduce_arcs + arcs_reduce

        input_length = self.num_players
        encoder_hidden = self.encoder.initHidden()

        encoder_outputs = torch.zeros(self.num_players, self.input_batch_size,self.encoder.hidden_size, device=device)

        for ei in range(input_length):

            encoder_output, encoder_hidden = self.encoder(encoder_input[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0]

        SOS_token = self.num_arc_class-1


        decoder_attention_list=[]

        self.player_out_list = []

        for player_id in selected_player_list:


            decoder_input = torch.tensor([[SOS_token] for _ in range(self.input_batch_size)], device=device)

            out_list = []

            decoder_hidden = torch.mean(encoder_hidden, 0,keepdim=True)

            for di in range(self.num_ops):
                decoder_output, decoder_hidden, decoder_attention = self.decoder_list[player_id](
                    decoder_input, decoder_hidden, encoder_outputs)

                out_list.append(decoder_output)
                topv, topi = decoder_output.data.topk(1,-1)



                for batch_id in range(self.input_batch_size):

                    for arc_id in range(self.num_arc_class):
                        if arc_id == topi[batch_id].item():

                            self.curr_player_choices[batch_id][arc_id][player_id][di] = 1
                        else:
                            self.curr_player_choices[batch_id][arc_id][player_id][di] = 0


                decoder_input = topi.squeeze().detach()


            self.player_out_list.append(out_list)



        return encoder_input

    #in the beginning of evaluation phase, sample the architecture set in this iteration
    def generate_player_arcs(self,current_player_choices, selected_player_list):
        selected_player_arcs_normal = []
        selected_player_arcs_reduce = []


        for player_id in selected_player_list:
            for op_id in range(self.num_ops):
                for batch_id in range(self.input_batch_size):
                    for arc_class_id in range(self.num_arc_class - 1):
                        alpha_normal = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))

                        alpha_reduce = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))


                        temp_arc_choices = current_player_choices[batch_id][arc_class_id]
                        index = 0

                        # normal
                        for curr_node in range(2, self.num_nodes + 2):
                            candidate_list = []
                            uncandidate_list = []
                            flag = 0
                            for prev_node in range(curr_node):
                                for tmp_op in range(self.num_ops):
                                    if index == player_id and op_id == tmp_op:
                                        alpha_normal[index][tmp_op] = 1
                                        flag = 1
                                    elif temp_arc_choices[index][tmp_op] == 1:
                                        candidate_list.append((index, tmp_op))
                                    else:
                                        uncandidate_list.append((index, tmp_op))

                                index += 1
                            if flag == 0:
                                if len(candidate_list) >= 2:
                                    [prev_node1, prev_node2] = np.random.choice(len(candidate_list), size=2,
                                                                              replace=False)

                                    index_1, op_1 = candidate_list[prev_node1]
                                    index_2, op_2 = candidate_list[prev_node2]

                                elif len(candidate_list) == 1:
                                    index_1, op_1 = candidate_list[0]

                                    [prev_node2] = np.random.choice(len(uncandidate_list), size=1)

                                    index_2, op_2 = uncandidate_list[prev_node2]

                                else:

                                    [prev_node1, prev_node2] = np.random.choice(len(uncandidate_list), size=2,
                                                                              replace=False)

                                    index_1, op_1 = uncandidate_list[prev_node1]
                                    index_2, op_2 = uncandidate_list[prev_node2]

                                alpha_normal[index_1][op_1] = 1
                                alpha_normal[index_2][op_2] = 1
                            else:
                                if len(candidate_list) >= 1:
                                    [prev_node1] = np.random.choice(len(candidate_list), size=1)
                                    index_1, op_1 = candidate_list[prev_node1]
                                else:
                                    [prev_node1] = np.random.choice(len(uncandidate_list), size=1)
                                    index_1, op_1 = uncandidate_list[prev_node1]

                                alpha_normal[index_1][op_1] = 1

                        selected_player_arcs_normal.append(alpha_normal)

                        # reduce
                        for curr_node in range(2, self.num_nodes + 2):
                            candidate_list = []
                            uncandidate_list = []
                            flag = 0
                            for prev_node in range(curr_node):
                                for tmp_op in range(self.num_ops):
                                    if index == player_id and op_id == tmp_op:
                                        alpha_reduce[index - self.half_num_players][tmp_op] = 1
                                        flag = 1
                                    elif temp_arc_choices[index][tmp_op] == 1:
                                        candidate_list.append((index, tmp_op))
                                    else:
                                        uncandidate_list.append((index, tmp_op))

                                index += 1
                            if flag == 0:
                                if len(candidate_list) >= 2:
                                    [prev_node1, prev_node2] = np.random.choice(len(candidate_list), size=2,
                                                                              replace=False)

                                    index_1, op_1 = candidate_list[prev_node1]
                                    index_2, op_2 = candidate_list[prev_node2]

                                elif len(candidate_list) == 1:
                                    index_1, op_1 = candidate_list[0]

                                    [prev_node2 ]= np.random.choice(len(uncandidate_list), size=1)

                                    index_2, op_2 = uncandidate_list[prev_node2]

                                else:

                                    [prev_node1, prev_node2] = np.random.choice(len(uncandidate_list), size=2,
                                                                              replace=False)

                                    index_1, op_1 = uncandidate_list[prev_node1]
                                    index_2, op_2 = uncandidate_list[prev_node2]

                                alpha_reduce[index_1 - self.half_num_players][op_1] = 1
                                alpha_reduce[index_2 - self.half_num_players][op_2] = 1
                            else:
                                if len(candidate_list) >= 1:
                                    [prev_node1] = np.random.choice(len(candidate_list), size=1)
                                    index_1, op_1 = candidate_list[prev_node1]
                                else:
                                    [prev_node1] = np.random.choice(len(uncandidate_list), size=1)
                                    index_1, op_1 = uncandidate_list[prev_node1]

                                alpha_reduce[index_1 - self.half_num_players][op_1] = 1

                        selected_player_arcs_reduce.append(alpha_reduce)


        return selected_player_arcs_normal, selected_player_arcs_reduce

    #evaluation phase
    def compute_reward(self,current_player_choices,selected_player_list,epoch=0):
        selected_player_arcs_normal, selected_player_arcs_reduce = self.generate_player_arcs(current_player_choices,selected_player_list)


        final_selected_arcs_normal = []
        final_selected_arcs_reduce = []

        for arc in self.old_normal_arcs:
            if arc != None:
                final_selected_arcs_normal.append(arc)

        for arc in self.old_reduce_arcs:
            if arc != None:
                final_selected_arcs_reduce.append(arc)

        final_selected_arcs_normal = final_selected_arcs_normal + selected_player_arcs_normal
        final_selected_arcs_reduce = final_selected_arcs_reduce + selected_player_arcs_reduce


        for _ in range(1):

            self.MODEL.train(final_selected_arcs_normal,final_selected_arcs_reduce,epoch)

            epoch += 1

        logging.info("start valid {} arcs!".format(len(final_selected_arcs_reduce)))

        valid_acc_list = self.MODEL.infer(final_selected_arcs_normal,final_selected_arcs_reduce)

        mean_valid_acc_list = mean(valid_acc_list)


        players_reward_list = []
        valid_index = 0
        player_old_reward_list = np.zeros( [self.num_players,self.num_ops,self.input_batch_size,2] )
        old_arcs_reward_list = np.zeros( [self.input_batch_size,self.num_arc_class] )

        old_arc_index = 0
        for player_id in selected_player_list:
            for ops in range(self.num_ops):
                for batch_id in range(self.input_batch_size):
                    if self.old_normal_arcs[old_arc_index] != None:

                        player_old_reward_list[player_id][ops][batch_id][0] = valid_acc_list[valid_index] - mean_valid_acc_list


                        valid_index += 1

                        player_old_reward_list[player_id][ops][batch_id][1] = valid_acc_list[
                                                                                  valid_index] - mean_valid_acc_list

                        valid_index += 1



                    old_arc_index += 2

        for batch_id in range(self.input_batch_size):
            for arc_class_id in range(self.num_arc_class-1):
                old_arcs_reward_list[batch_id][arc_class_id] = valid_acc_list[valid_index] - mean_valid_acc_list

                valid_index += 1

        for player_id in selected_player_list:
            one_player_reward = []
            for ops in range(self.num_ops):
                reward = np.zeros( [self.input_batch_size,self.num_arc_class] )
                for batch_id in range(self.input_batch_size):
                    for arc_class_id in range(self.num_arc_class - 1):
                        reward[batch_id][arc_class_id] = valid_acc_list[valid_index] - mean_valid_acc_list

                        reward[batch_id][arc_class_id] = (reward[batch_id][arc_class_id]-old_arcs_reward_list[batch_id][arc_class_id]) - (player_old_reward_list[player_id][ops][batch_id][0]-player_old_reward_list[player_id][ops][batch_id][1] )
                        valid_index += 1
                    reward[batch_id][self.num_arc_class-1] = 0 - (player_old_reward_list[player_id][ops][batch_id][0]-player_old_reward_list[player_id][ops][batch_id][1] )
                one_player_reward.append(reward)
            players_reward_list.append(one_player_reward)

        rank_list = np.argsort(-np.array(valid_acc_list)).tolist()
        valid_acc_list = [valid_acc_list[r] for r in rank_list]
        final_selected_arcs_normal = [final_selected_arcs_normal[r] for r in rank_list]
        final_selected_arcs_reduce = [final_selected_arcs_reduce[r] for r in rank_list]



        return valid_acc_list,final_selected_arcs_normal,final_selected_arcs_reduce,players_reward_list


    #training phase
    def train_players(self,selected_player_list,players_reward_list,encoder_input,player_training_step_per_iter=200,epoch=0):


        start_time = time.time()






        for player_train_step in range(player_training_step_per_iter):

            self.encoder_optimizer.zero_grad()
            selected_player_index = np.random.randint(0,len(selected_player_list))
            selected_player_id = selected_player_list[selected_player_index]

            self.decoder_optimizer_list[selected_player_id].zero_grad()

            input_length = self.num_players
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.num_players, self.input_batch_size, self.encoder.hidden_size,
                                          device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(encoder_input[ei],
                                                              encoder_hidden)
                encoder_outputs[ei] += encoder_output[0]

            SOS_token = self.num_arc_class - 1

            decoder_attention_list = []






            decoder_input = torch.tensor([[SOS_token] for _ in range(self.input_batch_size)], device=device)  # SOS



            decoder_hidden = torch.mean(encoder_hidden, 0, keepdim=True)

            loss = 0
            for di in range(self.num_ops):
                decoder_output, decoder_hidden, decoder_attention = self.decoder_list[selected_player_id](
                        decoder_input, decoder_hidden, encoder_outputs)

                loss += self.criterion(decoder_output,players_reward_list[selected_player_index][di])
                topv, topi = decoder_output.data.topk(1, -1)




                decoder_input = topi.squeeze().detach()


            loss.backward()

            nn.utils.clip_grad_norm(self.decoder_list[selected_player_id].parameters(), self.args.grad_clip)
            nn.utils.clip_grad_norm(self.encoder.parameters(), self.args.grad_clip)
            self.encoder_optimizer.step()


            self.decoder_optimizer_list[selected_player_id].step()


            if player_train_step % 10 == 0:
                curr_time = time.time()
                log_string = ""
                log_string += "epoch={:<6d}".format(epoch)
                log_string += "ch_step={:<6d}".format(player_train_step)
                log_string += " loss={:<8.6f}".format(loss.item())
                log_string += " lr={:<8.4f}".format(self.args.player_learning_rate)
                log_string += " encoder |g|={:<8.4f}".format(utils.compute_grad_norm(self.encoder))
                log_string += " decoder {:<6d} |g|={:<8.4f}".format(selected_player_id,utils.compute_grad_norm(self.decoder_list[selected_player_id]))

                log_string += " mins={:<10.2f}".format(float(curr_time - start_time) / 60)
                logging.info(log_string)

    #the inference phase after training. If not in the last iteration, return encoder_input.
    #Otherwise, return the attention weights.
    def player_infer(self,selected_player_list,encoder_input = None,is_final = False):
        with torch.no_grad():
            if encoder_input == None:
                encoder_input = np.zeros([self.num_players, self.input_batch_size, self.num_ops])
                for player_id in range(self.num_players):
                    for batch_id in range(self.input_batch_size):
                        for op_id in range(self.num_ops):
                            selected_class = np.random.choice(self.num_arc_class)
                            encoder_input[player_id][batch_id][op_id] = selected_class
                encoder_input=torch.tensor(encoder_input)


            input_length = self.num_players
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.num_players, self.input_batch_size, self.encoder.hidden_size,
                                          device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(encoder_input[ei],
                                                              encoder_hidden)
                encoder_outputs[ei] += encoder_output[0]

            SOS_token = self.num_arc_class-1

            decoder_attention_list = []


            num_selected_players = len(selected_player_list)
            if is_final:
                assert num_selected_players == self.num_players
                player_attentions = torch.zeros(self.num_players,self.num_ops,self.input_batch_size,self.num_players)

            for player_id in selected_player_list:


                decoder_input = torch.tensor([[SOS_token] for _ in range(self.input_batch_size)], device=device)  # SOS

                out_list = []

                decoder_hidden = torch.mean(encoder_hidden, 0,keepdim=True)
                for di in range(self.num_ops):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder_list[player_id](
                        decoder_input, decoder_hidden, encoder_outputs)
                    if is_final:
                        player_attentions[player_id][di] = decoder_attention.data
                    out_list.append(decoder_output)  # size = batchsize,num_classes
                    topv, topi = decoder_output.data.topk(1, -1)



                    for batch_id in range(self.input_batch_size):
                        for arc_id in range(self.num_arc_class):
                            if arc_id == topi[batch_id].item():
                                encoder_input[player_id][batch_id][di] = arc_id
                                self.curr_player_choices[batch_id][arc_id][player_id][di] = 1
                            else:
                                self.curr_player_choices[batch_id][arc_id][player_id][di] = 0



                    decoder_input = topi.squeeze().detach()


        if is_final:
            return torch.mean(player_attentions,-2)
        else:
            return encoder_input




    def train_search(self):

        encoder_input = None


        current_normal_arc = []
        current_reduce_arc = []

        current_vl_acc_list = []

        start_time = time.time()

        self.class_distribution_list = []
        with torch.autograd.set_detect_anomaly(True):
            for iter_id in range(self.args.num_iterations):
                selected_player_list = self.select_players()

                logging.info("selected_players: {}".format(selected_player_list))

                encoder_input = self.sample(selected_player_list, encoder_input)


                logging.info("start training WS model!")
                valid_acc_list, selected_player_arcs_normal, selected_player_arcs_reduce, players_reward_list = self.compute_reward(
                    self.curr_player_choices, selected_player_list, iter_id)

                logging.info("start training players")
                self.train_players(selected_player_list, players_reward_list,encoder_input, self.args.player_training_step_per_iter,
                                   iter_id)

                current_normal_arc += selected_player_arcs_normal
                current_reduce_arc += selected_player_arcs_reduce
                current_vl_acc_list += valid_acc_list

                rank_list = np.argsort(-np.array(current_vl_acc_list)).tolist()
                current_vl_acc_list = [current_vl_acc_list[r] for r in rank_list[:10]]

                current_normal_arc = [current_normal_arc[r] for r in rank_list[:10]]
                current_reduce_arc = [current_reduce_arc[r] for r in rank_list[:10]]

                with open(os.path.join(self.args.GNAS_arc_dir, 'best_arch_pool_{}'.format(iter_id)), 'w') as fa_latest:
                    with open(os.path.join(self.args.GNAS_arc_dir, 'best_arch_acc_{}'.format(iter_id)),
                              'w') as fp_latest:
                        for index in range(10):
                            arch_acc = current_vl_acc_list[index]
                            arch = self.MODEL.model.genotype(current_normal_arc[index], current_reduce_arc[index])


                            fa_latest.write('{}\n'.format(arch))
                            fa_latest.write('{normal:}\n')
                            fa_latest.write('{}\n'.format(current_normal_arc[index]))
                            fa_latest.write('{reduce:}\n')
                            fa_latest.write('{}\n'.format(current_reduce_arc[index]))
                            fp_latest.write('{}\n'.format(arch_acc))

                utils.save(self.encoder, os.path.join(self.args.model_path, 'encoder.pt'))

                for player_id in selected_player_list:
                    utils.save(self.decoder_list[player_id],
                               os.path.join(self.args.model_path, 'decoder_{}.pt'.format(player_id)))

                curr_time = time.time()
                logging.info("iter {0} ---------- time {1}".format(iter_id, (curr_time - start_time) / 60))


                encoder_input = self.player_infer(selected_player_list, encoder_input)







                #统计每个batch里面，聚类分布：
                #self.curr_player_choices #shape = batch_size,class,num_players,ops

                #record the number of players in each group in each iteration
                logging.info('total_players: {}'.format(torch.sum(self.curr_player_choices)  ))
                player_choices = torch.sum(self.curr_player_choices,dim=-1)
                player_choices = torch.sum(player_choices,dim = -1)#.view(self.num_arc_class,-1)

                num_players_in_arc_classes,_ = player_choices[:,:self.num_arc_class-1].topk(self.num_arc_class-1,-1,True,True)
                logging.info('--num_players_in_arc_classes {}'.format(num_players_in_arc_classes))
                num_players_in_arc_classes = torch.mean(num_players_in_arc_classes,dim=0)

                num_players_not_in_arc_classes = player_choices[:,self.num_arc_class-1]
                logging.info('num_players_not_in_arc_classes {}'.format( num_players_not_in_arc_classes ))
                num_players_not_in_arc_classes = torch.mean(num_players_not_in_arc_classes)

                logging.info("{}".format(num_players_in_arc_classes))
                logging.info("{}".format(num_players_in_arc_classes.numpy().tolist() ))
                logging.info("{}".format(num_players_not_in_arc_classes.item() ))

                self.class_distribution_list.append( num_players_in_arc_classes.numpy().tolist()  + [num_players_not_in_arc_classes.item()]  )

                logging.info( "iter_id = {0}, class_distribution = {1}".format( iter_id,self.class_distribution_list[-1] ) )



                if iter_id == self.args.num_iterations - 1:

                    #draw the distribution of players
                    color_list = [ 'r','m', 'y','g','b','k' ]
                    marker_list = [ 'o', 'v','^','<','>','s' ]

                    plt.xlabel('num_iterations')
                    plt.ylabel('num_players')
                    my_x_ticks = np.arange(0, self.args.num_iterations+1, 20)

                    plt.xticks(my_x_ticks)


                    tmp_iter_indices = [i for i in range(1,self.args.num_iterations+1)]
                    for tmp_class_id in range(self.num_arc_class):

                        tmp_class_distribution = [self.class_distribution_list[tmp_iter_index][tmp_class_id]  for tmp_iter_index in range(self.args.num_iterations)]
                        if tmp_class_id == self.num_arc_class - 1:
                            plt.plot(tmp_iter_indices, tmp_class_distribution, color=color_list[-1],
                                     linestyle='-', marker=marker_list[tmp_class_id], linewidth=1,
                                     label="class {}".format(tmp_class_id))
                        else:

                            plt.plot(tmp_iter_indices,tmp_class_distribution,color = color_list[tmp_class_id], linestyle='-', marker = marker_list[tmp_class_id], linewidth=1,label = "class {}".format(tmp_class_id))

                    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))

                    pic_save_path = "{0}/{1}".format(self.args.GNAS_arc_dir,'distribution.png')
                    plt.savefig(pic_save_path, dpi=120)

                    #return the attention weights
                    player_attentions = self.player_infer([player_id for player_id in range(self.num_players)],encoder_input,is_final=True  )
                    _,player_prefers = player_attentions.topk(10,-1,True,True)


                    return player_prefers.numpy()



    #the improved random search algorithm
    def RS_plus(self,player_prefers,arcs_per_iteration=500):

        def sample_one_arc(player_prefers):
            flag_list = [0 for _ in range(self.num_players)]

            alpha_normal = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))

            alpha_reduce = torch.tensor(np.zeros([self.half_num_players, self.num_ops]))

            #normal
            start_index = 0
            for curr_node in range(self.num_nodes):
                end_index= start_index + curr_node + 2
                max_flag = max( flag_list[start_index:end_index] )
                cand_player_ops_list = []
                for prev_node in range(curr_node+2):
                    if flag_list[start_index] == max_flag:
                        for op_id in range(self.num_ops):
                            cand_player_ops_list.append( (start_index,op_id) )

                    start_index +=1
                assert len(cand_player_ops_list) > 1

                [ prev_node1, prev_node2 ] = np.random.choice(len(cand_player_ops_list),size=2,replace=False )
                index1,op1 = cand_player_ops_list[prev_node1]
                index2,op2 = cand_player_ops_list[prev_node2]

                for pref_player in player_prefers[index1][op1]:
                    flag_list[pref_player] += 1
                for pref_player in player_prefers[index2][op2]:
                    flag_list[pref_player] += 1


                alpha_normal[index1][op1] = 1
                alpha_normal[index2][op2] = 1


            #reduce
            for curr_node in range(self.num_nodes):
                end_index= start_index + curr_node + 2
                max_flag = max( flag_list[start_index:end_index] )
                cand_player_ops_list = []
                for prev_node in range(curr_node+2):
                    if flag_list[start_index] == max_flag:
                        for op_id in range(self.num_ops):
                            cand_player_ops_list.append( (start_index,op_id) )

                    start_index +=1
                assert len(cand_player_ops_list) > 1

                [ prev_node1, prev_node2 ] = np.random.choice(len(cand_player_ops_list),size=2,replace=False )
                index1,op1 = cand_player_ops_list[prev_node1]
                index2,op2 = cand_player_ops_list[prev_node2]

                for pref_player in player_prefers[index1][op1]:
                    flag_list[pref_player] += 1
                for pref_player in player_prefers[index2][op2]:
                    flag_list[pref_player] += 1


                index1 = index1 - self.half_num_players
                index2 = index2 - self.half_num_players
                alpha_reduce[index1][op1] = 1
                alpha_reduce[index2][op2] = 1


            return alpha_normal,alpha_reduce

        current_normal_arc = []
        current_reduce_arc = []

        current_vl_acc_list = []

        start_time = time.time()
        with torch.autograd.set_detect_anomaly(True):
            for iter_id in range(self.args.num_iterations):

                sampled_normal_arc = []
                sampled_reduce_arc = []

                for _ in range(arcs_per_iteration):
                    alpha_normal,alpha_reduce = sample_one_arc(player_prefers)
                    sampled_normal_arc.append(alpha_normal)
                    sampled_reduce_arc.append(alpha_reduce)



                logging.info("start training RS_WS model!")

                epoch = iter_id
                for _ in range(1):
                    self.RS_MODEL.train(sampled_normal_arc, sampled_reduce_arc, epoch)

                    epoch += 1

                logging.info("start valid {} arcs!".format(len(sampled_normal_arc)))

                valid_acc_list = self.RS_MODEL.infer(sampled_normal_arc, sampled_reduce_arc)




                current_normal_arc += sampled_normal_arc
                current_reduce_arc += sampled_reduce_arc
                current_vl_acc_list += valid_acc_list

                rank_list = np.argsort(-np.array(current_vl_acc_list)).tolist()
                current_vl_acc_list = [current_vl_acc_list[r] for r in rank_list[:10]]

                current_normal_arc = [current_normal_arc[r] for r in rank_list[:10]]
                current_reduce_arc = [current_reduce_arc[r] for r in rank_list[:10]]

                with open(os.path.join(self.args.RS_plus_arc_dir, 'best_arch_pool_{}'.format(iter_id)), 'w') as fa_latest:
                    with open(os.path.join(self.args.RS_plus_arc_dir, 'best_arch_acc_{}'.format(iter_id)),
                              'w') as fp_latest:
                        for index in range(10):
                            arch_acc = current_vl_acc_list[index]
                            arch = self.MODEL.model.genotype(current_normal_arc[index], current_reduce_arc[index])


                            fa_latest.write('{}\n'.format(arch))
                            fa_latest.write('{normal:}\n')
                            fa_latest.write('{}\n'.format(current_normal_arc[index]))
                            fa_latest.write('{reduce:}\n')
                            fa_latest.write('{}\n'.format(current_reduce_arc[index]))
                            fp_latest.write('{}\n'.format(arch_acc))





                curr_time = time.time()
                logging.info("iter {0} ---------- time {1}".format(iter_id, (curr_time - start_time) / 60))


