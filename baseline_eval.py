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
            lattice_name = lattice_path.split('/')[-1]
            target_name = os.path.join(target_dir, lattice_name)
            #utils.check_file(target_name)
            print('target_name: {}'.format(target_name))
            if os.path.isfile(os.path.join(target_name)):
                print('In if')
                target_path_list.append(target_name)
                lattice_path_list.append(lattice_path)

    return lattice_path_list, target_path_list

def load_pred(path):
        a = np.load(path)
        return a['edge_data'][:,-1].tolist()

def load_ref(path):
    a = np.load(path)
    return a['target'].tolist()

def load_eval_data(lattice_path_list, target_path_list):
    preds = []
    refs = []
    for pred_path, ref_path in zip(lattice_path_list, target_path_list):
        pred = load_pred(pred_path)
        preds.append(pred)
        ref = load_ref(ref_path)
        refs.append(ref)
    return np.array(preds), np.array(refs)

def main(args):

    lattice_path_list, target_path_list = pred_ref_lists(args.file_split, args.target_dir)
    preds, refs = load_eval_data(lattice_path_list, target_path_list)

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
    description = "Run evaluation on baseline numbers"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-t', '--target-dir', required=True, type=str,
                        help='Path to the target directory containing *.npz target files')
    parser.add_argument('-f', '--file-split', required=True, type=str,
                        help='Path to the file containing paths to all processed lattices in the test set.')
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
