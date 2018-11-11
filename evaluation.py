"""Evaluation functions for LatticeRNN."""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def log(array):
    """Clip the elements of arrays before taking log."""
    return np.log(np.clip(array, a_min=1e-8, a_max=1.0))

def nce(labels, predictions):
    """Calculate normalised cross entropy.

    Arguments:
        labels {array} -- numpy array of labels {0, 1}
        predictions {array} -- numpy array of predictions, [0, 1]

    Returns:
        float -- NCE score
    """
    assert len(labels) == len(predictions), \
           "dimension of labels is not the same as predictions"
    percentage_correct = np.sum(labels) / len(labels)
    label_entropy = - percentage_correct * log(percentage_correct) \
                    - (1 - percentage_correct) * log(1 - percentage_correct)
    conditional_entropy = (np.dot(labels, log(predictions)) \
                           + np.dot((1 - labels), log(1 - predictions))) \
                          / (-len(labels))
    score = (label_entropy - conditional_entropy) / label_entropy
    return score

def roc(labels, predictions):
    """Compute ROC curve and its AUC.

    Arguments:
        labels {array} -- numpy array of labels {0, 1}
        predictions {array} -- numpy array of predictions, [0, 1]

    Returns:
        tuple -- fpr array, tpr array, area float
    """
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = auc(fpr, tpr)
    return fpr, tpr, area

def pr(labels, predictions):
    """Compute precision-recall curve and its AUC.

    Arguments:
        labels {array} -- numpy array of labels {0, 1}
        predictions {array} -- numpy array of predictions, [0, 1]

    Returns:
        tuple -- precision array, recall array, area float
    """
    precision, recall, _ = precision_recall_curve(labels, predictions)
    area = auc(recall, precision)
    return precision, recall, area

def plot_roc(fpr, tpr, area, name, dst_dir):
    """Plotting ROC curve.

    Arguments:
        tpr {list} -- a list of numpy 1D array for true positive rate
        fpr {list} -- a list of numpy 1D array for false positive rate
        area {list} -- a list of floats for area under curve
        name {str} -- text for the legend
        dst_dir {str} -- output figure directory, file name `roc.pdf`
    """
    plt.clf()
    plt.figure(figsize=(3, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    for (x_val, y_val, a_val, string) in zip(fpr, tpr, area, name):
        label = string + ' (AUC = %0.4f)' %a_val
        plt.plot(x_val, y_val, label=label)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(dst_dir, 'roc.pdf'), bbox_inches='tight')
    plt.close()

def plot_pr(precision, recall, area, name, dst_dir):
    """Plotting ROC curve.

    Arguments:
        tpr {list} -- a list of numpy 1D array for true positive rate
        fpr {list} -- a list of numpy 1D array for false positive rate
        area {list} -- a list of floats for area under curve
        name {str} -- text for the legend
        dst_dir {str} -- output figure directory, file name `roc.pdf`
    """
    plt.clf()
    plt.figure(figsize=(3, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    for (x_val, y_val, a_val, string) in zip(recall, precision, area, name):
        label = string + ' (AUC = %0.4f)' %a_val
        plt.plot(x_val, y_val, label=label)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(os.path.join(dst_dir, 'pr.pdf'), bbox_inches='tight')
    plt.close()

def plot_det(fnr, fpr, name, dst_dir):
    """Plotting DET curve.

    Arguments:
        fnr {list} -- a list of numpy 1D array for true positive rate
        fpr {list} -- a list of numpy 1D array for false positive rate
        name {str} -- text for the legend
        dst_dir {str} -- output figure directory, file name `det.pdf`
    """
    plt.clf()
    plt.figure(figsize=(3, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    for (x_val, y_val, string) in zip(fpr, fnr, name):
        plt.plot(x_val, y_val, label=string)
    plt.legend(loc='upper right')
    plt.xlim([0.01, 1])
    plt.ylim([0.01, 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('False Negative Rate')
    plt.xlabel('False Positive Rate')
    axes = plt.gca()
    axes.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    axes.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.savefig(os.path.join(dst_dir, 'det.pdf'), bbox_inches='tight')
