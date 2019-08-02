import argparse
import evaluation
import numpy as np
import os
import sys
import utils


def pred_ref_lists(file_split, target_dir):
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

def load_pred(path, indices):
    """ Load the posterior predictions for the one-best. """
    a = np.load(path)
    return a['edge_data'][indices,-1].tolist()

def load_eval_data(lattice_path_list, target_path_list):
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
    split_file = os.path.join(args.root, 'test.txt')
    lattice_path_list, target_path_list = pred_ref_lists(split_file, target_dir)
    preds, refs = load_eval_data(lattice_path_list, target_path_list)
    assert len(preds) == len(refs), \
         'Predictions and references must be sequences of the same length'

    print(preds)
    print(refs)
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
    """ Parse the command line arguments.
    """
    description = "Run evaluation on the one-best baseline numbers"
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
