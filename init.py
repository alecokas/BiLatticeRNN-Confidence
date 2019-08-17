"""Model setup."""

import os
import torch
import utils, lstm


def setup(opt, checkpoint):
    """Create new model or reload from checkpoint model"""
    # Resume model if checkpoint is provided, create a new model otherwise.
    if checkpoint is not None:
        model_path = os.path.join(opt.resume, checkpoint['model_file'])
        utils.check_file(model_path)
        print("".ljust(4) + "=> Resuming model from %s" %model_path)
        model = torch.load(model_path)
    else:
        print("".ljust(4) + "=> Creating new model")
        model = lstm.create_model(opt)

    # Load optim_file if checkpoint is provided, return None otherwise.
    if checkpoint is not None:
        optim_path = os.path.join(opt.resume, checkpoint['optim_file'])
        utils.check_file(optim_path)
        print("".ljust(4) + "=> Resuming optim_state from %s" %optim_path)
        optim_state = torch.load(optim_path)
    else:
        optim_state = None

    return model, optim_state
