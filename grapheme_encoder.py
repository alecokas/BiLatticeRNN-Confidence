""" `grapheme_encoder.py` defines:
    * Grapheme merging attention scheme
    * Optional BiDirectional RNN, GRU, or LSTM encoder
"""


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init


class LuongAttention(torch.nn.Module):
    """ Luong attention layer as defined in: https://arxiv.org/pdf/1508.04025.pdf 
        Specifically defined with grapheme combination in mind.
    """
    def __init__(self, attn_type, num_features, initialisation):
        """ Initialise the Attention layer 
        
            Arguments:
                attn_type {string}: The type of attention similarity function to apply
                num_features {int}: The number of input feature dimensions per grapheme
                initialisation {string}: The type of weight initialisation to use
        """
        super(LuongAttention, self).__init__()
        self.num_features = num_features
        self.attn_type = attn_type
        self.initialisation = initialisation
        self.use_bias = True

        if self.attn_type not in ['dot', 'mult', 'concat', 'scaled-dot', 'concat-enc-key']:
            raise ValueError(self.attn_type, "is not an appropriate attention type.")

        if self.attn_type == 'mult':
            self.attn = torch.nn.Linear(self.num_features, self.num_features, self.use_bias)
            self.initialise_parameters()
        elif self.attn_type == 'concat':
            self.attn = torch.nn.Linear(self.num_features * 2, self.num_features, self.use_bias)
            self.v = Variable(torch.randn(self.num_features))
            self.initialise_parameters()
        elif self.attn_type == 'concat-enc-key':
            # TODO: This assumes the added information is 1D - generalise this
            self.attn = torch.nn.Linear(self.num_features * 2 + 1, self.num_features, self.use_bias)
            self.v = Variable(torch.randn(self.num_features))
            self.initialise_parameters()

    def dot_score(self, key, query):
        """ Dot product similarity function """
        return torch.sum(key * query, dim=2)

    def mult_score(self, key, query):
        """ Multiplicative similarity function (also called general) """
        energy = self.attn(query)
        return torch.sum(key * energy, dim=2)

    def concat_score(self, key, query):
        """ Concatinative similarity function (also called additive) """
        energy = self.attn(torch.cat((key.expand(query.size(0), -1, -1), query), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, key, query, val):
        """ Compute and return the attention weights and the result of the weighted sum.
            key, query, val are of the tensor form: (Arcs, Graphemes, Features)
        """
        # Calculate the attention weights (alpha) based on the given attention type
        if self.attn_type == 'mult':
            attn_energies = self.mult_score(key, query)
        elif self.attn_type == 'concat':
            attn_energies = self.concat_score(key, query)
        elif self.attn_type == 'dot':
            attn_energies = self.dot_score(key, query)
        elif self.attn_type == 'scaled-dot':
            attn_energies = self.dot_score(key, query) / self.num_features
        elif self.attn_type == 'concat-enc-key':
            attn_energies = self.concat_score(key, query)

        # Alpha is the softmax normalized probability scores (with added dimension)
        alpha = F.softmax(attn_energies, dim=1).unsqueeze(1)
        # The context is the result of the weighted summation
        context = torch.bmm(alpha, val)
        return context, alpha

    def initialise_parameters(self):
        """Initialise parameters for all layers."""
        init_method = getattr(init, self.initialisation)
        init_method(self.attn.weight.data)
        if self.use_bias:
            init.constant(self.attn.bias.data, val=0)

class GraphemeEncoder(nn.Module):
    """ Bi-directional recurrent neural network designed 
        to encode a grapheme feature sequence.
    """
    def __init__(self, opt):
        nn.Module.__init__(self)

        # Defining some parameters
        self.hidden_size = opt.grapheme_hidden_size
        self.num_layers = opt.grapheme_num_layers
        self.initialisation = opt.init_grapheme
        self.use_bias = True

        if opt.encoder_type == 'RNN':
            self.encoder = nn.RNN(
                input_size=opt.grapheme_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=opt.grapheme_bidirectional,
                batch_first=True,
                dropout=opt.encoding_dropout,
                bias=True
            )
        elif opt.encoder_type == 'LSTM':
            self.encoder = nn.LSTM(
                input_size=opt.grapheme_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=opt.grapheme_bidirectional,
                batch_first=True,
                dropout=opt.encoding_dropout,
                bias=True
            )
        elif opt.encoder_type == 'GRU':
            self.encoder = nn.GRU(
                input_size=opt.grapheme_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=opt.grapheme_bidirectional,
                batch_first=True,
                dropout=opt.encoding_dropout,
                bias=True
            )
        else:
            raise ValueError('Unexpected encoder type: Got {} but expected RNN, GRU, or LSTM'.format(opt.encoder_type))


        self.initialise_parameters()

    def forward(self, x):
        """ Passing in the input into the model and obtaining outputs and the updated hidden state """
        out, hidden_state = self.encoder(x)
        return out, hidden_state

    def init_hidden_state(self, batch_size):
        """ Generate the first hidden state of zeros """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def initialise_parameters(self):
        """ Initialise parameters for all layers. """
        init_method = getattr(init, self.initialisation)
        init_method(self.encoder.weight_ih_l0.data)
        init_method(self.encoder.weight_hh_l0.data)
        if self.use_bias:
            init.constant(self.encoder.bias_ih_l0.data, val=0)
            init.constant(self.encoder.bias_hh_l0.data, val=0)