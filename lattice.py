"""
    `lattice.py` defines:
    * Lattice object containing node and edge information - optional subword level information
    * An in-place reverse function.
"""

import sys
import numpy as np


class Lattice:
    """ Lattice object """

    def __init__(self, path, word_mean=None, word_std=None, subword_mean=None, subword_std=None, lattice_type='grapheme'):
        """ Lattice object.

            Arguments:
                path: A string absolute path of a pre-processed lattice
                word_mean: A NumPy array containing the mean vector
                           of the word information in the dataset.
                word_std: A NumPy array containing the standard deviation
                          of the word information in the dataset.
                subword_mean: A NumPy array containing the mean vector
                              of the subword information in the dataset.
                subword_std: A NumPy array containing the standard deviation
                             of the subword information in the dataset.
                lattice_type: A string which indicates if the model is expecting
                              `word` or `grapheme` lattices
        """
        self.path = path
        self.word_mean = word_mean
        self.word_std = word_std
        self.grapheme_mean = subword_mean
        self.grapheme_std = subword_std
        self.child_dict = None
        self.parent_dict = None
        self.edges = None
        self.mask = None
        self.ignore = []
        self.is_grapheme = True if lattice_type == 'grapheme' else False
        self.load()

    def load(self):
        """ Load the pre-processed lattice.
            Normalise to zero mean and unit variance if mean and std are provided.
        """
        data = np.load(self.path)
        self.nodes = list(data['topo_order'])
        self.edges = data['edge_data']
        self.child_dict = data['child_2_parent'].item()
        self.parent_dict = data['parent_2_child'].item()

        if self.is_grapheme:
            self.grapheme_data = data['grapheme_data']

        # Backward compatibility
        try:
            self.ignore = list(data['ignore'])
        except KeyError:
            pass

        self.node_num = len(self.nodes)
        self.edge_num = self.edges.shape[0]

        if self.edge_num > 0:
            self.edges = self.normalise(self.edges, self.word_mean, self.word_std)
        else:
            raise Exception('All lattices must have a definite positive number of nodes.')

        if self.is_grapheme:
            self.grapheme_data = self.normalise(self.grapheme_data, self.grapheme_mean, self.grapheme_std)

    def normalise(self, x, mean, std):
        """ Apply data whitening to x """
        if mean is not None and std is not None:
            if mean.shape[1] == x.shape[1] and std.shape[1] == x.shape[1]:
                return (x - mean) / std
            else:
                raise Exception("Dimension of mean and std vector is inconsistent with data.")

    def reverse(self):
        """ Reverse the graph """
        self.nodes.reverse()
        self.child_dict, self.parent_dict = self.parent_dict, self.child_dict

    def feature_dim(self):
        return self.edges.shape[1]

class Target:
    """ Target object """

    def __init__(self, path):
        """ Target constructor

            Arguments:
                path {str}: absolute path to target file.
        """
        self.path = path
        self.target = None
        self.indices = None
        self.ref = None
        self.load()

    def load(self):
        """ Load target, one-best path indices and reference """
        data = np.load(self.path)
        self.target = data['target']
        self.indices = list(data['indices'])
        self.ref = list(data['ref'])
