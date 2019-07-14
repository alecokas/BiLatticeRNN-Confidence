"""Plot training and validation losses."""

import os
import matplotlib.pyplot as plt
import numpy as np
import utils

def plot(directory, onebest=False):
    """Plot training and validation cross entropy."""
    plt.clf()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16

    cols = [2] if onebest else [1, 2]
    train_file = os.path.join(directory, 'train.log')
    val_file = os.path.join(directory, 'val.log')
    utils.check_file(train_file)
    utils.check_file(val_file)
    train_history = np.loadtxt(train_file, usecols=cols)
    val_history = np.loadtxt(val_file, usecols=cols)
    assert train_history.size == val_history.size

    x_axis = list(range(1, train_history.shape[0] + 1))
    axis = plt.gca()
    if onebest:
        plt.plot(x_axis, train_history, 'r-', label='train')
        plt.plot(x_axis, val_history, 'g-', label='val')
    else:
        plt.plot(x_axis, train_history[:, 0], 'r-', label='train_allarc')
        plt.plot(x_axis, val_history[:, 0], 'g-', label='val_allarc')
        # plt.plot(x_axis, train_history[:, 1], 'r--', label='train_onebest')
        # plt.plot(x_axis, val_history[:, 1], 'g--', label='val_onebest')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(x_axis)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Cross Entropy')
    plt.savefig(os.path.join(directory, 'plot.pdf'), bbox_inches='tight')
    plt.close()
