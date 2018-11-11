"""`dataloader.py` defines:
    * a customized dataset object for lattices
    * a function to create dataloaders for train, val, test
"""

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils, lattice

class LatticeDataset(Dataset):
    """Lattice dataset."""

    def __init__(self, data_file, stats_file, tgt_dir, percentage):
        """Load data file and dataset statistics."""
        self.data_file = data_file
        self.tgt_dir = tgt_dir
        self.percentage = percentage
        self.data = []
        self.target = []

        np.random.seed(1)
        with open(self.data_file, 'r') as file_in:
            for line in file_in:
                line = line.strip()
                if line:
                    utils.check_file(line)
                    tgt_path = os.path.join(self.tgt_dir, line.split('/')[-1])
                    utils.check_file(tgt_path)
                    if np.random.rand() < percentage:
                        self.data.append(line)
                        self.target.append(tgt_path)
                    else:
                        pass
        stats = np.load(stats_file)
        self.mean = stats['mean']
        self.std = stats['std']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (lattice.Lattice(self.data[idx], self.mean, self.std),
                lattice.Target(self.target[idx]))

def collate_fn(batch):
    """Collate data and target of each item in the batch in lists."""
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

def create(opt):
    """Create DataLoader object for each set."""
    loaders = []
    stats_file = os.path.join(opt.data, opt.dataset, 'stats.npz')
    utils.check_file(stats_file)
    tgt_dir = os.path.join(opt.data, opt.dataset, 'target')
    utils.check_dir(tgt_dir)
    if opt.debug:
        print("".ljust(4) + "=> Creating data loader for train.")
        data_file = os.path.join(opt.data, opt.dataset, 'train_debug.txt')
        utils.check_file(data_file)
        dataset = LatticeDataset(data_file, stats_file, tgt_dir, opt.trainPctg)
        loaders.append(DataLoader(dataset=dataset, batch_size=opt.batchSize,
                                  shuffle=opt.shuffle, collate_fn=collate_fn,
                                  num_workers=opt.nThreads))
        return loaders[0], None, None

    for split in ['train', 'val', 'test']:
        print("".ljust(4) + "=> Creating data loader for %s." %split)
        data_file = os.path.join(opt.data, opt.dataset, '%s.txt' %split)
        utils.check_file(data_file)
        dataset = LatticeDataset(data_file, stats_file, tgt_dir, opt.trainPctg)
        shuffle = False if split == 'test' else opt.shuffle
        loaders.append(DataLoader(dataset=dataset, batch_size=opt.batchSize,
                                  shuffle=shuffle, collate_fn=collate_fn,
                                  num_workers=opt.nThreads))
    return loaders[0], loaders[1], loaders[2]

def resample_dataset(opt, split):
    """Resampling from the entire dataset."""
    data_file = os.path.join(opt.data, opt.dataset, '%s.txt' %split)
    utils.check_file(data_file)
    stats_file = os.path.join(opt.data, opt.dataset, 'stats.npz')
    utils.check_file(stats_file)
    tgt_dir = os.path.join(opt.data, opt.dataset, 'target')
    utils.check_dir(tgt_dir)
    dataset = LatticeDataset(data_file, stats_file, tgt_dir, opt.trainPctg)
    loader = DataLoader(dataset=dataset, batch_size=opt.batchSize,
                        shuffle=opt.shuffle, collate_fn=collate_fn,
                        num_workers=opt.nThreads)
    return loader
