#!/usr/bin/env python3
"""Main function for training and testing."""

import os
import sys
import numpy as np
import torch
from torch.nn import functional as F
import opts, init, checkpoints, criterion, train, dataloader
import utils, evaluation, plot

def main():
    """Main function for training and testing."""
    # Parse command line arguments and cache
    opt = opts.Opts().args
    utils.savecmd(opt.resume, sys.argv)

    utils.print_color_msg("==> Setting up data loader")
    train_loader, val_loader, test_loader = dataloader.create(opt)

    # Load checkpoint if specified, None otherwise
    utils.print_color_msg("==> Checking checkpoints")
    checkpoint = checkpoints.load(opt)

    utils.print_color_msg("==> Setting up model and criterion")
    model, optim_state = init.setup(opt, checkpoint)
    loss_fn = criterion.setup(opt, checkpoint)

    utils.print_color_msg("==> Loading trainer")
    trainer = train.create_trainer(model, loss_fn, opt, optim_state)

    best_loss = float('Inf')
    val_loss = float('Inf')
    start_epoch = max([1, opt.epochNum])
    if checkpoint is not None:
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print("".ljust(4) + "Previous best loss: "
              + utils.color_msg('%.5f' %best_loss))

    if opt.valOnly:
        assert start_epoch > 1, "There must be at least one epoch"
        utils.print_color_msg("==> Validation:")
        print("".ljust(4) + "=> Epoch %i" %(start_epoch-1))
        trainer.val(val_loader, start_epoch-1)
        sys.exit()

    if opt.testOnly:
        assert start_epoch > 1, "There must be at least one epoch"
        utils.print_color_msg("==> Testing:")
        print("".ljust(4) + "=> Epoch %i" %(start_epoch-1))
        _, prediction, reference, post = trainer.test(test_loader, start_epoch-1)
        # TODO: For now we assume this is true: if opt.loss == 'BCELogit':
        prediction = F.sigmoid(torch.Tensor(prediction)).numpy()
        nce = evaluation.nce(reference, prediction)
        precision, recall, area = evaluation.pr(reference, prediction)
        precision_bl, recall_bl, area_bl = evaluation.pr(reference, post)
        utils.print_color_msg(
            "".ljust(7) + "NCE: %.4f. AUC(PR): %.4f. AUC(BL): %.4f" \
            %(nce, area, area_bl))
        trainer.logger['test'].write('NCE: %f\nAUC(PR): %f\n' %(nce, area))
        evaluation.plot_pr([precision, precision_bl], [recall, recall_bl],
                           [area, area_bl], ['BiLatticeRNN', 'posterior'],
                           opt.resume)
        np.savez(os.path.join(opt.resume, 'result.npz'),
                 prediction=prediction, reference=reference, posteriors=post)
        sys.exit()

    utils.print_color_msg("==> Training:")
    for epoch in range(start_epoch, opt.nEpochs+1):
        print("".ljust(4) + "=> Epoch %i" %epoch)
        best_model = False
        _ = trainer.train(train_loader, epoch, val_loss)

        if not opt.debug:
            val_loss = trainer.val(val_loader, epoch)
            if val_loss < best_loss:
                best_model = True
                print("".ljust(4) + "** Best model: "
                      + utils.color_msg('%.4f' %val_loss))
                best_loss = val_loss
            checkpoints.save(epoch, trainer.model, loss_fn,
                             trainer.optim_state, best_model, val_loss, opt)

    if not opt.debug:
        utils.print_color_msg("==> Testing:")
        _, prediction, reference, _ = trainer.test(test_loader, opt.nEpochs)
        prediction = F.sigmoid(torch.Tensor(prediction)).numpy()
        nce = evaluation.nce(reference, prediction)
        precision, recall, area = evaluation.pr(reference, prediction)
        utils.print_color_msg(
            "".ljust(7) + "NCE: %.4f. AUC(PR): %.4f" %(nce, area))
        trainer.logger['test'].write('NCE: %f\nAUC(PR): %f\n' %(nce, area))
        evaluation.plot_pr([precision], [recall], [area], ['BiLatticeRNN'], opt.resume)

        # Flush write out and reset pointer
        for open_file in trainer.logger.values():
            open_file.flush()
            open_file.seek(0)
        plot.plot(opt.resume, opt.onebest)

if __name__ == '__main__':
    main()
