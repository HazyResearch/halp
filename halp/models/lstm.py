import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from halp.utils.utils import single_to_half_det, single_to_half_stoc
from halp.utils.utils import copy_layer_weights, copy_module_weights
from halp.utils.utils import void_cast_func, get_recur_attr
from halp.layers.bit_center_layer import BitCenterModule
from halp.layers.linear_layer import BitCenterLinear
from halp.layers.cross_entropy import BitCenterCrossEntropy
from halp.layers.sigmoid_layer import BitCenterSigmoid
from halp.layers.tanh_layer import BitCenterTanh
from halp.layers.embedding import BitCenterEmbedding
from halp.layers.ele_mult import BitCenterEleMult
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('')


class BitCenterLSTMCell(BitCenterModule, nn.LSTMCell):
    '''
    Implementation of the LSTM cell
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 bias=True,
                 cast_func=void_cast_func,
                 n_train_sample=1):
        BitCenterModule.__init__(self)
        # nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.cast_func = cast_func
        self.n_train_sample = n_train_sample

        # we process the input and last hidden state in a batch for the 4 gates
        self.input_linear = BitCenterLinear(
            input_size,
            hidden_size * 4,
            bias=self.bias,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        self.hidden_linear = BitCenterLinear(
            hidden_size,
            hidden_size * 4,
            bias=self.bias,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        # for the naming of the symbols like i, f, g, o, please refer to
        # https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        self.i_activation = BitCenterSigmoid(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.f_activation = BitCenterSigmoid(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.g_activation = BitCenterTanh(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.o_activation = BitCenterSigmoid(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)

        self.f_c_mult = BitCenterEleMult(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.i_g_mult = BitCenterEleMult(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)

        self.c_prime_activation = BitCenterTanh(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.o_c_prime_mult = BitCenterEleMult(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)

    def forward(self, x, state):
        c, h = state
        trans_input = self.input_linear(x)
        trans_hidden = self.hidden_linear(h)

        out = trans_input + trans_hidden
        i = self.i_activation(out[:, 0:self.hidden_size])
        f = self.f_activation(out[:, self.hidden_size:(2 * self.hidden_size)])
        g = self.g_activation(
            out[:, (2 * self.hidden_size):(3 * self.hidden_size)])
        o = self.o_activation(out[:, 3 * self.hidden_size:])
        c_prime = self.f_c_mult(f, c) + self.i_g_mult(i, g)
        c_prime_act = self.c_prime_activation(c_prime)
        h_prime = self.o_c_prime_mult(o, c_prime_act)

        return (h_prime, c_prime)

def copy_lstm_cell_weights(src, tar):
    # source is bitcenter LSTM cell, tar is the conventional
    tar.weight_ih.data.copy_(src.input_linear.weight)
    tar.weight_hh.data.copy_(src.hidden_linear.weight)
    tar.bias_ih.data.copy_(src.input_linear.bias)
    tar.bias_hh.data.copy_(src.hidden_linear.bias)
    return tar

def copy_lstm_weights_to_non_bc_lstm_cell(src, tar):
    tar.weight_ih.data.copy_(src.weight_ih_l0)
    tar.weight_hh.data.copy_(src.weight_hh_l0)
    tar.bias_ih.data.copy_(src.bias_ih_l0)
    tar.bias_hh.data.copy_(src.bias_hh_l0)
    return tar

def copy_lstm_weights_to_bc_lstm_cell(src, tar):
    tar.input_linear.weight.data.copy_(src.weight_ih_l0)
    tar.hidden_linear.weight.data.copy_(src.weight_hh_l0)
    tar.input_linear.bias.data.copy_(src.bias_ih_l0)
    tar.hidden_linear.bias.data.copy_(src.bias_hh_l0)
    return tar

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # self.hidden = self.init_hidden()

    def init_hidden(self, sentence):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        batch_size = sentence.size(1)
        return (torch.zeros(1, batch_size, self.hidden_dim, 
                dtype=self.word_embeddings.weight.dtype,
                device=self.word_embeddings.weight.device),
                torch.zeros(1, batch_size, self.hidden_dim,
                dtype=self.word_embeddings.weight.dtype,
                device=self.word_embeddings.weight.device))

    def forward(self, sentence):
        self.hidden = self.init_hidden(sentence)
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out, self.hidden = self.lstm(
        #     embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(-1, self.hidden_dim))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space


class BitCenterLSTMTagger(BitCenterModule):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 hidden_size,
                 cast_func=void_cast_func,
                 n_classes=10,
                 n_train_sample=1,
                 dtype="bc"):
        BitCenterModule.__init__(self)
        self.cast_func = cast_func
        self.n_train_sample = n_train_sample
        self.n_classes = n_classes
        self.dtype = dtype
        self.embedding = BitCenterEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.lstm_cell = BitCenterLSTMCell(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bias=True,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.linear = BitCenterLinear(
            in_features=hidden_size,
            out_features=n_classes,
            bias=True,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.criterion = BitCenterCrossEntropy(
            cast_func=cast_func, n_train_sample=n_train_sample)

        if dtype == "bc":
            pass
        elif (dtype == "fp") or (dtype == "lp"):
            self.embedding = copy_layer_weights(
                self.embedding,
                nn.Embedding(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim))
            self.lstm_cell = copy_lstm_cell_weights(
                self.lstm_cell,
                nn.LSTMCell(
                    embedding_dim, hidden_size, bias=self.lstm_cell.bias))
            self.linear = copy_layer_weights(
                self.linear,
                nn.Linear(in_features=hidden_size, out_features=n_classes))
            self.criterion = nn.CrossEntropyLoss(size_average=True)
            if dtype == "lp":
                if self.cast_func == void_cast_func:
                    pass
                else:
                    for child in self.children():
                        child.half()
        else:
            raise Exception(dtype + " is not supported in LeNet!")

    def init_hidden(self, x):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        batch_size = x.size(1)
        seq_length = x.size(0)
        dtype = self.embedding.weight.dtype
        device = self.embedding.weight.device
        hidden_dim = self.lstm_cell.hidden_size
        return (torch.zeros(seq_length + 1, batch_size, hidden_dim, dtype=dtype, device=device),
                torch.zeros(seq_length + 1, batch_size, hidden_dim, dtype=dtype, device=device))

    def forward(self, x, y, test=False):
        # we assume the first dimension corresponds to steps
        # The second dimension corresponds to sample index
        (c, h) = self.init_hidden(x)
        out = self.embedding(x)
        for i in range(out.size(0)):
            state = self.lstm_cell(out[i, :, :], (c[i], h[i]))
            c[i + 1].data.copy_(state[0])
            h[i + 1].data.copy_(state[1])
        out = self.linear(c[1:])
        self.output = out
        out = out.view(-1, out.size(-1))
        if test:
            return out
        else:
            # print(out.size(), y.size())
            self.loss = self.criterion(out, y)
            if isinstance(self.criterion, BitCenterCrossEntropy) \
                and self.criterion.do_offset == False:
                self.output = self.output + self.criterion.input_lp
            return self.loss

    def predict(self, x):
        output = self.forward(x, y=None, test=True)
        pred = output.data.cpu().numpy().argmax(axis=1)
        return pred, output
