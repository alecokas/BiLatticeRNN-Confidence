import argparse
import evaluation
import numpy as np
import os
import sys
import utils


def pred_ref_lists(file_split, target_dir):
    """ Obtain the predictions and reference confidence score lists.

        Arguments:
            file_split: The file containing the paths to processed lattices in the test set
            target_dir: Name of the target directory
    """
    with open(file_split, 'r') as test_split_file:
        lattice_path_list = []
        target_path_list = []
        for lattice_path in test_split_file:
            lattice_path = lattice_path.strip('\n')
            lattice_name = lattice_path.split('/')[-1]
            target_name = os.path.join(target_dir, lattice_name)
            if os.path.isfile(target_name):
                target_path_list.append(target_name)
                lattice_path_list.append(lattice_path)

    return lattice_path_list, target_path_list

def load_ref(path):
    """ Load the references and arc indices for the one-best. """
    a = np.load(path)
    return a['ref'].tolist(), a['indices'].tolist()

def load_pred(path, indices, post_idx=-1):
    """ Load the posterior predictions for the one-best.
        By default we assume that the posteriors are stored in the final column.

        Arguments:
            path: Path to processed lattice file (*.npz)
            indices: The arc indices to obtain posteriors for
            post_idx: Posterior column index
    """
    a = np.load(path)
    return a['edge_data'][indices, post_idx].tolist()

def load_eval_data(lattice_path_list, target_path_list):
    """ Generate a numpy array of predictions and an array of references.
    """
    preds = []
    refs = []
    for pred_path, ref_path in zip(lattice_path_list, target_path_list):
        ref, indices = load_ref(ref_path)
        refs = refs + ref
        pred = load_pred(pred_path, indices)
        preds = preds + pred
    return np.array(preds), np.array(refs)

def main(args):
    target_dir = os.path.join(args.root, args.target_dir)
    test_split_file = os.path.join(args.root, 'test.txt')
    lattice_path_list, target_path_list = pred_ref_lists(test_split_file, target_dir)
    preds, refs = load_eval_data(lattice_path_list, target_path_list)
    assert len(preds) == len(refs), \
         'Predictions and references must be sequences of the same length'

    nce = evaluation.nce(refs, preds)
    precision, recall, area = evaluation.pr(refs, preds)
    precision_bl, recall_bl, area_bl = evaluation.pr(refs, preds)
    utils.print_color_msg(
        "".ljust(7) + "NCE: %.4f. AUC(PR): %.4f. AUC(BL): %.4f" \
        %(nce, area, area_bl)
    )
    print('NCE: %f\nAUC(PR): %f\n' %(nce, area))
    evaluation.plot_pr(
        [precision, precision_bl], [recall, recall_bl],
        [area, area_bl], ['BiRNN', 'posterior']
    )


def parse_arguments(args_to_parse):
    """ Parse the command line arguments. """
    description = "Run evaluation on the one-best baseline numbers for word posterior confidence estimator. "
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-r', '--root', required=True, type=str,
                        help='Path to the directory containing the test.txt file with paths to all processed lattices in the test set.')
    parser.add_argument('-t', '--target-dir', required=True, type=str,
                        help='Path to the target directory containing *.npz target files - relative to root')
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
