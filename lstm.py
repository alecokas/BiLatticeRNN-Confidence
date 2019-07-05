"""`LSTM.py` defines:
    * basic LSTM cell,
    * LSTM layers for lattices building from LSTM cell,
    * DNN layers including the output layer,
    * LatticeRNN model that connects LSTM layers and DNN layers.
"""

import math
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
from utils import Dimension

class LSTMCell(nn.LSTMCell):
    """Overriding initialization and naming methods of LSTMCell."""

    def reset_parameters(self):
        """Orthogonal Initialization."""
        init.orthogonal(self.weight_ih.data)
        self.weight_hh.data.set_(torch.eye(self.hidden_size).repeat(4, 1))
        # The bias is just set to zero vectors.
        if self.bias:
            init.constant(self.bias_ih.data, val=0)
            init.constant(self.bias_hh.data, val=0)

    def __repr__(self):
        """Rename."""
        string = '{name}({input_size}, {hidden_size})'
        if 'bias' in self.__dict__ and self.bias is False:
            string += ', bias={bias}'
        return string.format(name=self.__class__.__name__, **self.__dict__)

class LSTM(nn.Module):
    """A module that runs LSTM through the lattice."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, bidirectional=True, attention=None, **kwargs):
        """Build multi-layer LSTM from LSTM cell."""
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.attention = attention

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 \
                    else hidden_size * self.num_directions
                cell = cell_class(input_size=layer_input_size,
                                  hidden_size=hidden_size,
                                  bias=use_bias, **kwargs)
                suffix = '_reverse' if direction == 1 else ''
                setattr(self, 'cell_{}{}'.format(layer, suffix), cell)
        self.reset_parameters()

    def get_cell(self, layer, direction):
        """Get LSTM cell by layer."""
        suffix = '_reverse' if direction == 1 else ''
        return getattr(self, 'cell_{}{}'.format(layer, suffix))

    def reset_parameters(self):
        """Initialise parameters for all cells."""
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                cell = self.get_cell(layer, direction)
                cell.reset_parameters()

    def combine_edges(self, method, lattice, hidden, in_edges):
        """Methods for combining hidden states of all incoming edges.

        Arguments:
            method {str} -- choose from 'max', 'mean', 'posterior', 'attention', 'attention_simple'
            lattice {obj} -- lattice object
            hidden {list} -- each element is a hidden representation
            in_edges {list} -- each element is the index of incoming edges

        Raises:
            NotImplementedError -- the method is not yet implemented

        Returns:
            tensor -- the combined hidden representation
        """
        # Default: last element of the input feature is the posterior prob
        index = -1
        in_hidden = torch.cat([hidden[i].view(1, -1) for i in in_edges], 0)
        if len(in_edges) == 1:
            return in_hidden
        elif method == 'max':
            posterior = torch.cat([lattice.edges[i, index] for i in in_edges])
            _, max_idx = torch.max(posterior, 0)
            result = in_hidden[max_idx]
            return result
        elif method == 'mean':
            result = torch.mean(in_hidden, 0, keepdim=True)
            return result
        elif method == 'posterior':
            posterior = torch.cat([lattice.edges[i, index] for i in in_edges])
            posterior = posterior*lattice.std[0, index] + lattice.mean[0, index]
            posterior.data.clamp_(min=1e-6)
            posterior = posterior/torch.sum(posterior)
            if np.isnan(posterior.data.numpy()).any():
                print("posterior is nan")
                sys.exit(1)
            result = torch.mm(posterior.view(1, -1), in_hidden)
            return result
        elif method == 'attention':
            assert self.attention is not None, "build attention model first."
            # Posterior of incoming edges
            posterior = torch.cat([lattice.edges[i, index] for i in in_edges]).view(-1, 1)
            # Undo whitening
            posterior = posterior * lattice.std[0, index] + lattice.mean[0, index]
            context = torch.cat(
                (posterior, torch.ones_like(posterior) * torch.mean(posterior),
                 torch.ones_like(posterior)*torch.std(posterior)), dim=1)
            weights = self.attention.forward(in_hidden, context)
            result = torch.mm(weights, in_hidden)
            return result
        else:
            raise NotImplementedError

    def _forward_rnn(self, cell, lattice, input_, method, state):
        """Forward through one layer of LSTM."""
        edge_hidden = [None] * lattice.edge_num
        node_hidden = [None] * lattice.node_num

        edge_cell = [None] * lattice.edge_num
        node_cell = [None] * lattice.node_num

        node_hidden[lattice.nodes[0]] = state[0].view(1, -1)
        node_cell[lattice.nodes[0]] = state[1].view(1, -1)
        # The incoming and outgoing edges must be:
        # either a list of lists (for confusion network)
        # or a list of ints (for normal lattices)
        for each_node in lattice.nodes:
            # If the node is a child, compute its node state by combining all
            # the incoming edge states.
            if each_node in lattice.child_dict:
                in_edges = [i for i in lattice.child_dict[each_node].values()]
                if all(isinstance(item, list) for item in in_edges):
                    in_edges = [item for sublist in in_edges
                                for item in sublist]
                else:
                    assert all(isinstance(item, int) for item in in_edges)
                node_hidden[each_node] = self.combine_edges(
                    method, lattice, edge_hidden, in_edges)
                node_cell[each_node] = self.combine_edges(
                    method, lattice, edge_cell, in_edges)

            # If the node is a parent, compute each outgoing edge states
            if each_node in lattice.parent_dict:
                out_edges = lattice.parent_dict[each_node].values()
                if all(isinstance(item, list) for item in out_edges):
                    out_edges = [item for sublist in out_edges
                                 for item in sublist]
                else:
                    assert all(isinstance(item, int) for item in out_edges)
                for each_edge in out_edges:
                    old_state = (node_hidden[each_node], node_cell[each_node])
                    if each_edge in lattice.ignore:
                        new_state = old_state
                    else:
                        new_state = cell.forward(input_[each_edge].view(1, -1),
                                                 old_state)
                    edge_hidden[each_edge], edge_cell[each_edge] = new_state

        end_node_state = (node_hidden[lattice.nodes[-1]],
                          node_cell[lattice.nodes[-1]])
        edge_hidden = torch.cat(edge_hidden, 0)
        return edge_hidden, end_node_state

    def forward(self, lattice, method):
        """Complete multi-layer LSTM network."""
        # Set initial states to zero
        h_0 = Variable(lattice.edges.data.new(self.num_directions,
                                              self.hidden_size).zero_())
        state = (h_0, h_0)
        output = lattice.edges
        if self.bidirectional:
            lattice.reverse()
        for layer in range(self.num_layers):
            cur_output, cur_h_n, cur_c_n = [], [], []
            for direction in range(self.num_directions):
                cell = self.get_cell(layer, direction)
                cur_state = (state[0][direction], state[1][direction])
                if self.bidirectional:
                    lattice.reverse()
                layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    self, cell=cell, lattice=lattice, input_=output,
                    method=method, state=cur_state)
                cur_output.append(layer_output)
                cur_h_n.append(layer_h_n)
                cur_c_n.append(layer_c_n)
            output = torch.cat(cur_output, 1)
            cur_h_n = torch.cat(cur_h_n, 0)
            cur_c_n = torch.cat(cur_c_n, 0)
            state = (cur_h_n, cur_c_n)
        return output

class DNN(nn.Module):
    """A module that defines multi-layer fully connected neural networks."""

    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 initialization, use_bias=True, logit=False):
        """Build multi-layer FC."""
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.initialization = initialization
        self.use_bias = use_bias
        self.logit = logit

        if num_layers > 0:
            for layer in range(num_layers):
                layer_input_size = input_size if layer == 0 else hidden_size
                fc = nn.Linear(layer_input_size, hidden_size, bias=use_bias)
                setattr(self, 'fc_{}'.format(layer), fc)
            self.out = nn.Linear(hidden_size, output_size, bias=use_bias)
        else:
            self.out = nn.Linear(input_size, output_size, bias=use_bias)
        self.reset_parameters()

    def get_fc(self, layer):
        """Get FC layer by layer number."""
        return getattr(self, 'fc_{}'.format(layer))

    def reset_parameters(self):
        """Initialise parameters for all layers."""
        init_method = getattr(init, self.initialization)
        for layer in range(self.num_layers):
            fc = self.get_fc(layer)
            init_method(fc.weight.data)
            if self.use_bias:
                init.constant(fc.bias.data, val=0)
        init_method(self.out.weight.data)
        init.constant(self.out.bias.data, val=0)

    def forward(self, x):
        """Complete multi-layer DNN network."""
        for layer in range(self.num_layers):
            fc = self.get_fc(layer)
            x = F.relu(fc(x))
        output = self.out(x)
        if self.logit:
            return output
        else:
            return F.sigmoid(output)

class Attention(nn.Module):
    """A module that defines multi-layer fully connected neural networks."""

    def __init__(self, input_size, hidden_size, num_layers,
                 initialization, use_bias=True):
        """Build multi-layer FC."""
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.initialization = initialization
        self.use_bias = use_bias

        if num_layers > 0:
            for layer in range(num_layers):
                layer_input_size = input_size if layer == 0 else hidden_size
                fc = nn.Linear(layer_input_size, hidden_size, bias=use_bias)
                setattr(self, 'attention_{}'.format(layer), fc)
            self.out = nn.Linear(hidden_size, 1, bias=use_bias)
        else:
            self.out = nn.Linear(input_size, 1, bias=use_bias)
        self.reset_parameters()

    def get_fc(self, layer):
        """Get FC layer by layer number."""
        return getattr(self, 'attention_{}'.format(layer))

    def reset_parameters(self):
        """Initialise parameters for all layers."""
        init_method = getattr(init, self.initialization)
        for layer in range(self.num_layers):
            fc = self.get_fc(layer)
            init_method(fc.weight.data)
            if self.use_bias:
                init.constant(fc.bias.data, val=0)
        init_method(self.out.weight.data)
        init.constant(self.out.bias.data, val=0)

    def forward(self, x, context):
        """Complete multi-layer DNN network."""
        # Concat context with hidden representations
        output = torch.cat((x, context), dim=1)
        for layer in range(self.num_layers):
            fc = self.get_fc(layer)
            output = F.relu(fc(output))
        output = self.out(output).view(1, -1)
        output = F.tanh(output)
        return F.softmax(output, dim=1)

class DotProdAttention(nn.Module):
    """ A class which defines the dot product attention mechanism. """

    def __init__(self, scale=True, dropout=0.1):
        """ Initialise the dot product attention mechanism """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def forward(self, query, key, value, mask=None):
        """ A forward pass of the attention memchanism which operates over the graphemes.
        
            query:  Tensor with dimensions: (Arc, Grapheme, Feature)
            key:    Tensor with dimensions: (Arc, Grapheme, Feature)
            value:  Tensor with dimensions: (Arc, Grapheme, Feature)
        """
        # Ensure that the key and query are the same length
        d_k = key.size(-1)
        assert query.size(-1) == d_k

        # Compute compatability function and normalise across the grapheme dimension
        # Query:            (Arc, Grapheme, Feature)
        # transpose(Key):   (Arc, Feature, Grapheme)
        # Weight Matrix:    (Arc, Feature, Feature)
        # Compatability fn: (Arc, Grapheme, Grapheme)
        # W = q A k'
        attention_weights = torch.bmm(query, key.transpose(Dimension.seq, Dimension.feature))

        if self.scale:
            attention_weights = attention_weights / math.sqrt(d_k)

        # Softmax normalisation of attention weights over the grapheme sequence
        # Zero pad attention weights
        attention_weights = torch.exp(attention_weights)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask, 0)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        # Apply dropout and weight value
        attention_weights = self.dropout(attention_weights)
        context = torch.bmm(attention_weights, value)
        return context


class Attention2(nn.Module):
    def __init__(self, input_size, initialization, use_bias=True):
        super(Attention2, self).__init__()
        self.input_size = input_size
        self.initialization = initialization
        self.use_bias = use_bias

        self.attention_layer = nn.Linear(input_size, input_size, bias=use_bias)
        self.context_vector = nn.Parameter(torch.zeros(input_size),
                                           requires_grad=True)

    def reset_parameters(self):
        init_method = getattr(init, self.initialization)
        init_method(self.context_vector)
        init_method(self.attention_layer.weight)
        init.constant(self.attention_layer.bias, val=0)

    def forward(self, x, extension):
        output = torch.cat((x, extension), dim=1)
        output = self.attention_layer(output)
        output = F.tanh(output)
        output = torch.matmul(output, self.context_vector)
        return F.softmax(output.view(1, -1), dim=1)

class Model(nn.Module):
    """Bidirectional LSTM model on lattices."""

    def __init__(self, opt):
        """Basic model building blocks."""
        nn.Module.__init__(self)
        self.opt = opt

        if self.opt.method == 'attention':
            self.attention = Attention(self.opt.hiddenSize + 3,
                                       self.opt.attentionSize,
                                       self.opt.attentionLayers, self.opt.init,
                                       use_bias=True)
        else:
            self.attention = None

        num_directions = 2 if self.opt.bidirectional else 1
        self.lstm = LSTM(LSTMCell, self.opt.inputSize, self.opt.hiddenSize,
                         self.opt.nLSTMLayers, use_bias=True,
                         bidirectional=self.opt.bidirectional,
                         attention=self.attention)

        self.dnn = DNN(num_directions * self.opt.hiddenSize,
                       self.opt.linearSize, 1, self.opt.nFCLayers,
                       self.opt.init, use_bias=True, logit=True)

    def forward(self, lattice):
        """Forward pass through the model."""
        # BiLSTM -> FC(relu) -> LayerOut(sigmoid if not logit)
        output = self.lstm.forward(lattice, self.opt.method)
        output = self.dnn.forward(output)
        return output

def create_model(opt):
    """New Model object."""
    model = Model(opt)
    model.share_memory()
    return model
