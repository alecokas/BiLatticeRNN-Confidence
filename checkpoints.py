"""Loading and saving checkpoints."""

import os
import subprocess
import torch
import utils

def load(opt):
    """Select checkpoint to load."""
    ckpt = opt.epochNum
    if ckpt == 0:
        print("".ljust(4) + "=> No checkpoint to load. Retrain the model.")
        return None
    else:
        if ckpt == -1:
            print("".ljust(4) + "=> Loading the latest checkpoint.")
            ckpt_path = os.path.join(opt.resume, 'latest.pth')
        elif ckpt == -2:
            print("".ljust(4) + "=> Loading the best checkpoint.")
            ckpt_path = os.path.join(opt.resume, 'best.pth')
        else:
            raise ValueError("Should not reach here!")
        utils.check_file(ckpt_path)
        return torch.load(ckpt_path)

def save(epoch, model, criterion, optim_state, best_model, loss, opt):
    """Save a checkpoint."""
    if opt.saveOne:
        cmd = ['rm -f']
        cmd.append(os.path.join(opt.resume, '/model_*.pth'))
        cmd.append(os.path.join(opt.resume, '/criterion_*.pth'))
        cmd.append(os.path.join(opt.resume, '/optim_stat_*.pth'))
        subprocess.call(' '.join(cmd), shell=True)

    model_file = 'model_%i.pth' %epoch
    criterion_file = 'criterion_%i.pth' %epoch
    optim_file = 'optim_state_%i.pth' %epoch
    torch.save(model, os.path.join(opt.resume, model_file))
    torch.save(criterion, os.path.join(opt.resume, criterion_file))
    torch.save(optim_state, os.path.join(opt.resume, optim_file))
    info = {'epoch':epoch, 'model_file':model_file,
            'criterion_file':criterion_file, 'optim_file':optim_file,
            'loss':loss}
    torch.save(info, os.path.join(opt.resume, 'latest.pth'))

    if best_model:
        info = {'epoch':epoch, 'model_file':model_file,
                'criterion_file':criterion_file, 'optim_file':optim_file,
                'loss':loss}
        torch.save(info, os.path.join(opt.resume, 'best.pth'))
        torch.save(model, os.path.join(opt.resume, 'model_best.pth'))
