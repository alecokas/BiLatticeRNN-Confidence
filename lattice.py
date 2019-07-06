"""`lattice.py` defines:
    * Lattice object containing node and edge information
    * A in-palce reverse function.
"""

import sys
import numpy as np

class Lattice:
    """Lattice object."""

    def __init__(self, path, mean=None, std=None):
        """Lattice object.

        Arguments:
            path {string} -- absolute path of a pre-processed lattice

        Keyword Arguments:
            mean {numpy array} -- mean vector of the dataset (default: {None})
            std {numpy array} -- standard deviation of the dataset (default:
                {None})
        """
        self.path = path
        self.mean = mean
        self.std = std
        self.child_dict = None
        self.parent_dict = None
        self.edges = None
        self.mask = None
        self.ignore = []
        self.load()

    def load(self):
        """Load the pre-processed lattice.
        Normalise to zero mean and unit variance if mean and std are provided.
        """
        data = np.load(self.path)
        self.nodes = list(data['topo_order'])
        self.edges = data['edge_data']
        self.child_dict = data['child_2_parent'].item()
        self.parent_dict = data['parent_2_child'].item()

        if 'grapheme_data' in data:
            self.grapheme_data = data['grapheme_data']

        # Backward compatibility
        try:
            self.ignore = list(data['ignore'])
        except KeyError:
            pass

        self.node_num = len(self.nodes)
        self.edge_num = self.edges.shape[0]

        if self.edge_num > 0:
            self.feature_dim = self.edges.shape[1]
            if self.mean is not None:
                if self.mean.shape[1] == self.feature_dim:
                    self.edges = self.edges - self.mean
                else:
                    print("Dimension of mean vector is inconsistent with data.")
                    sys.exit(1)
            if self.std is not None:
                if self.std.shape[1] == self.feature_dim:
                    self.edges = self.edges / self.std
                else:
                    print("Dimension of std vector is inconsistent with data.")
                    sys.exit(1)
        else:
            self.feature_dim = None

    def reverse(self):
        """Reverse the graph."""
        self.nodes.reverse()
        self.child_dict, self.parent_dict = self.parent_dict, self.child_dict

class Target:
    """Target object."""

    def __init__(self, path):
        """Target constructor

        Arguments:
            path {str} -- absolute path to target file.
        """
        self.path = path
        self.target = None
        self.indices = None
        self.ref = None
        self.load()

    def load(self):
        """Load target, one-best path indices and reference."""
        data = np.load(self.path)
        self.target = data['target']
        self.indices = list(data['indices'])
        self.ref = list(data['ref'])
