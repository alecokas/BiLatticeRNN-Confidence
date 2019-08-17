""" `LSTM.py` defines:
    * basic LSTM cell,
    * LSTM layers for lattices building from LSTM cell,
"""


import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init


DURATION_IDX = 50


class LSTMCell(nn.LSTMCell):
    """ Overriding initialization and naming methods of LSTMCell. """

    def reset_parameters(self):
        """ Orthogonal Initialization """
        init.orthogonal(self.weight_ih.data)
        self.weight_hh.data.set_(torch.eye(self.hidden_size).repeat(4, 1))
        # The bias is just set to zero vectors.
        if self.bias:
            init.constant(self.bias_ih.data, val=0)
            init.constant(self.bias_hh.data, val=0)

    def __repr__(self):
        """ Rename """
        string = '{name}({input_size}, {hidden_size})'
        if 'bias' in self.__dict__ and self.bias is False:
            string += ', bias={bias}'
        return string.format(name=self.__class__.__name__, **self.__dict__)

class LSTM(nn.Module):
    """A module that runs LSTM through the lattice."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, bidirectional=True, attention=None, **kwargs):
        """ Build multi-layer LSTM from LSTM cell """
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
        """ Get LSTM cell by layer. """
        suffix = '_reverse' if direction == 1 else ''
        return getattr(self, 'cell_{}{}'.format(layer, suffix))

    def reset_parameters(self):
        """ Initialise parameters for all cells. """
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                cell = self.get_cell(layer, direction)
                cell.reset_parameters()

    def combine_edges(self, combine_method, lattice, hidden, in_edges):
        """ Methods for combining hidden states of all incoming edges.

            Arguments:
                combine_method {str}: choose from 'max', 'mean', 'posterior', 'attention', 'attention_simple'
                lattice {obj}: lattice object
                hidden {list}: each element is a hidden representation
                in_edges {list}: each element is the index of incoming edges

            Raises:
                NotImplementedError: the method is not yet implemented

            Returns:
                tensor: the combined hidden representation
        """
        # Default: last element of the input feature is the posterior prob
        index = -1
        in_hidden = torch.cat([hidden[i].view(1, -1) for i in in_edges], 0)
        if len(in_edges) == 1:
            return in_hidden
        elif combine_method == 'max':
            posterior = torch.cat([lattice.edges[i, index] for i in in_edges])
            _, max_idx = torch.max(posterior, 0)
            result = in_hidden[max_idx]
            return result
        elif combine_method == 'mean':
            result = torch.mean(in_hidden, 0, keepdim=True)
            return result
        elif combine_method == 'posterior':
            posterior = torch.cat([lattice.edges[i, index] for i in in_edges])
            posterior = posterior*lattice.std[0, index] + lattice.mean[0, index]
            posterior.data.clamp_(min=1e-6)
            posterior = posterior/torch.sum(posterior)
            if np.isnan(posterior.data.numpy()).any():
                print("posterior is nan")
                sys.exit(1)
            result = torch.mm(posterior.view(1, -1), in_hidden)
            return result
        elif combine_method == 'attention':
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

    def _forward_rnn(self, cell, lattice, input_, combine_method, state):
        """ Forward through one layer of LSTM. """
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
                    combine_method, lattice, edge_hidden, in_edges)
                node_cell[each_node] = self.combine_edges(
                    combine_method, lattice, edge_cell, in_edges)

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

    def forward(self, lattice, combine_method):
        """ Complete multi-layer LSTM network. """
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
                    combine_method=combine_method, state=cur_state)
                cur_output.append(layer_output)
                cur_h_n.append(layer_h_n)
                cur_c_n.append(layer_c_n)
            output = torch.cat(cur_output, 1)
            cur_h_n = torch.cat(cur_h_n, 0)
            cur_c_n = torch.cat(cur_c_n, 0)
            state = (cur_h_n, cur_c_n)
        return output
