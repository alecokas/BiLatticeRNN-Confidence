"""`dataloader.py` defines:
    * a customized dataset object for lattices
    * a function to create dataloaders for train, val, test
"""

import logging
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import utils, lattice

class LatticeDataset(Dataset):
    """ Lattice dataset object.

        data:
            A pyhton list of paths to preprocessed lattices.

        target:
            A python list of paths to target files.
    """

    def __init__(self, data_file, stats_file, tgt_dir, percentage, lattice_type):
        """ Load data file and dataset statistics. """
        self.data_file = data_file
        self.tgt_dir = tgt_dir
        self.percentage = percentage
        self.data = []
        self.target = []
        self.lattice_type = lattice_type
        self.log_location = '/'.join(data_file.split('/')[:-1] + ['dataset.log'])

        np.random.seed(1)
        with open(self.data_file, 'r') as file_in:
            for line in file_in:
                line = line.strip()
                if line:
                    utils.check_file(line)
                    tgt_path = os.path.join(self.tgt_dir, line.split('/')[-1])

                    lattice_has_target = True
                    if not os.path.isfile(tgt_path):
                        logging.basicConfig(
                            filename=self.log_location, filemode='w',
                            format='%(asctime)s - %(message)s', level=logging.INFO
                        )
                        logging.info('Warning: {} cannot be found - skipping this lattice.'.format(tgt_path))
                        lattice_has_target = False

                    if np.random.rand() < percentage and lattice_has_target:
                        self.data.append(line)
                        self.target.append(tgt_path)
                    else:
                        pass
        stats = np.load(stats_file)
        self.word_mean = stats['mean']
        self.word_std = stats['std']
        if 'subword_mean' in stats and 'subword_std' in stats:
            self.subword_mean = stats['subword_mean']
            self.subword_std = stats['subword_std']
        else:
            self.subword_mean = None
            self.subword_std = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (lattice.Lattice(self.data[idx], self.word_mean, self.word_std, self.subword_mean, self.subword_std, lattice_type=self.lattice_type),
                lattice.Target(self.target[idx]))

def collate_fn(batch):
    """Collate data and target of each item in the batch in lists."""
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

def create(opt):
    """Create DataLoader object for each set."""
    loaders = []
    stats_file = os.path.join(opt.data, 'stats.npz')
    utils.check_file(stats_file)
    tgt_dir = os.path.join(opt.data, opt.target)
    utils.check_dir(tgt_dir)
    if opt.debug:
        print("".ljust(4) + "=> Creating data loader for train.")
        data_file = os.path.join(opt.data, 'train_debug.txt')
        utils.check_file(data_file)
        dataset = LatticeDataset(data_file, stats_file, tgt_dir, opt.trainPctg, opt.lattice_type)
        loaders.append(DataLoader(dataset=dataset, batch_size=opt.batchSize,
                                  shuffle=opt.shuffle, collate_fn=collate_fn,
                                  num_workers=opt.nThreads))
        return loaders[0], None, None
    if opt.subtrain:
        for split in ['subtrain', 'cv', 'test']:
            print("".ljust(4) + "=> Creating data loader for {}.".format(split))
            data_file = os.path.join(opt.data, '{}.txt'.format(split))
            utils.check_file(data_file)
            dataset = LatticeDataset(data_file, stats_file, tgt_dir, opt.trainPctg, opt.lattice_type)
            shuffle = False if split == 'test' else opt.shuffle
            loaders.append(DataLoader(dataset=dataset, batch_size=opt.batchSize,
                                    shuffle=shuffle, collate_fn=collate_fn,
                                    num_workers=opt.nThreads))
        return loaders[0], loaders[1], loaders[2]

    for split in ['train', 'cv', 'test']:
        print("".ljust(4) + "=> Creating data loader for %s." %split)
        data_file = os.path.join(opt.data, '%s.txt' %split)
        utils.check_file(data_file)
        dataset = LatticeDataset(data_file, stats_file, tgt_dir, opt.trainPctg, opt.lattice_type)
        shuffle = False if split == 'test' else opt.shuffle
        loaders.append(DataLoader(dataset=dataset, batch_size=opt.batchSize,
                                  shuffle=shuffle, collate_fn=collate_fn,
                                  num_workers=opt.nThreads))
    return loaders[0], loaders[1], loaders[2]

def resample_dataset(opt, split):
    """Resampling from the entire dataset."""
    data_file = os.path.join(opt.data, '%s.txt' %split)
    utils.check_file(data_file)
    stats_file = os.path.join(opt.data, 'stats.npz')
    utils.check_file(stats_file)
    tgt_dir = os.path.join(opt.data, opt.target)
    utils.check_dir(tgt_dir)
    dataset = LatticeDataset(data_file, stats_file, tgt_dir, opt.trainPctg, opt.lattice_type)
    loader = DataLoader(dataset=dataset, batch_size=opt.batchSize,
                        shuffle=opt.shuffle, collate_fn=collate_fn,
                        num_workers=opt.nThreads)
    return loader
