"""Criterion setup."""

import os
import torch
import torch.nn as nn
import utils

def setup(opt, checkpoint):
    """Setup criterion."""
    # Load criterion if checkpoint is provided, create one otherwise
    if checkpoint is not None:
        criterion_path = os.path.join(opt.resume, checkpoint['criterion_file'])
        utils.check_file(criterion_path)
        print("".ljust(4) + "=> Resuming criterion from %s" %criterion_path)
        criterion = torch.load(criterion_path)
    else:
        print("".ljust(4) + "=> Creating new criterion")
        criterion = create_criterion()

    return criterion

def create_criterion():
    """Create a new BCE with logit criterion."""
    return nn.BCEWithLogitsLoss(size_average=False)
