"""Training routines for LSTM model."""

import os
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm
import utils, criterion

class Trainer():
    """Trainer object for training, validation and testing the model."""

    def __init__(self, model, criterion, opt, optim_state):
        """Setup optim_state, optimizer and logger."""
        self.model = model
        self.criterion = criterion
        self.optim_state = optim_state
        self.opt = opt

        # Only set weight decay to weights
        params_dict = dict(self.model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if 'weight' in key and ('fc' in key or 'out' in key
                                    or 'attention' in key):
                params += [{'params': value, 'weight_decay': opt.weightDecay}]
            else:
                params += [{'params': value, 'weight_decay': 0.0}]

        # Set optimizer
        if opt.optimizer == 'SGD':
            self.optimizer = optim.SGD(params, lr=opt.LR,
                                       momentum=opt.momentum)
        elif opt.optimizer == 'Adam':
            self.optimizer = optim.Adam(params, lr=opt.LR,
                                        betas=(0.9, 0.999), eps=1e-8)
        else:
            raise NotImplementedError

        # Set new optim_state if retrain, restore if exist
        if self.optim_state is None:
            self.optim_state = {'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'epoch': -1, 'initial_lr': self.opt.LR}
            log_option = 'w+'
        else:
            self.model.load_state_dict(self.optim_state['state_dict'])
            self.optimizer.load_state_dict(self.optim_state['optimizer'])
            log_option = 'a+'
        self.optimizer.param_groups[0]['initial_lr'] = \
            self.optim_state['initial_lr']

        # Learning rate scheduler
        if opt.LRDecay == 'anneal':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=opt.LRDParam,
                last_epoch=self.optim_state['epoch'])
        elif opt.LRDecay == 'stepwise':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=3, gamma=opt.LRDParam,
                last_epoch=self.optim_state['epoch'])
        elif opt.LRDecay == 'newbob':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=opt.LRDParam, patience=1)
        else:
            assert opt.LRDecay == 'none'

        self.logger = \
            {'train': open(os.path.join(opt.resume, 'train.log'), log_option),
             'val': open(os.path.join(opt.resume, 'val.log'), log_option),
             'test': open(os.path.join(opt.resume, 'test.log'), log_option)}

    def xent_onebest(self, output, indices, reference):
        """Compute confidence score binary cross entropy."""
        assert len(indices) == len(reference), "inconsistent one-best sequence."
        loss, count = 0, 0
        pred_onebest, ref_onebest = [], []
        if indices:
            # Extract scores on one-best and apply sigmoid
            prediction = [output[i] for i in indices]
            for pred, ref in zip(prediction, reference):
                if ref is not None:
                    count += 1
                    pred_onebest.append(pred)
                    ref_onebest.append(ref)
            one_best_pred = Variable(torch.Tensor(pred_onebest))
            one_best_ref = Variable(torch.Tensor(ref_onebest))
            loss_fn = criterion.create_criterion()
            loss = loss_fn(one_best_pred, one_best_ref).data[0]
        return loss, count, pred_onebest, ref_onebest

    def xent(self, output, ignore, reference):
        """Compute confidence score binary cross entropy."""
        loss, count = 0, 0
        all_pred, all_ref = [], []
        for i, (pred, ref) in enumerate(zip(output, reference)):
            if i not in ignore:
                count += 1
                all_pred.append(pred)
                all_ref.append(float(ref))
        all_pred_t = Variable(torch.Tensor(all_pred))
        all_ref_t = Variable(torch.Tensor(all_ref))
        loss_fn = criterion.create_criterion()
        loss = loss_fn(all_pred_t, all_ref_t).data[0]
        return loss, count, all_pred, all_ref

    @staticmethod
    def mean(value, count):
        """Deal with zero count."""
        if count == 0:
            assert value == 0
            return 0
        else:
            return value / count

    @staticmethod
    def moving_average(avg, total_count, val, count):
        """Compute the weighted average"""
        all_val = avg * total_count + val
        all_counts = total_count + count
        return Trainer.mean(all_val, all_counts)

    def forward_one_lattice(self, lattice, target, index, results, update):
        """Forward through one single lattice on CPU."""
        if lattice.edge_num == 0 or not target.ref:
            results[index] = [(0, 0), (0, 0), ([], [])]
        else:
            if update:
                self.optimizer.zero_grad()
            lattice.edges = Variable(torch.from_numpy(lattice.edges).float())
            lattice.grapheme_data = Variable(torch.from_numpy(lattice.grapheme_data).float())
            target_t = Variable(
                torch.from_numpy(target.target).float().view(-1, 1))
            output = self.model.forward(lattice)
            target_length = target_t.size(0)

            # Error signals on all arcs but filter out arcs need to be ignored
            target_back = []
            output_back = []
            count = 0
            for i in range(target_length):
                if i not in lattice.ignore:
                    count += 1
                    target_back.append(target_t[i])
                    output_back.append(output[i])
            target_back = torch.cat(target_back).view(-1, 1)
            output_back = torch.cat(output_back).view(-1, 1)
            loss = self.criterion(output_back, target_back) / count

            # Error signals on one-best path
            assert len(target.indices) == len(target.ref), \
                   "inconsistent one-best sequence"
            target_back_onebest = []
            output_back_onebest = []
            count_onebest = 0
            pred_onebest, ref_onebest = [], []
            if target.indices:
                for j in target.indices:
                    count_onebest += 1
                    target_back_onebest.append(target_t[j])
                    output_back_onebest.append(output[j])
                    pred_onebest.append(output[j].data[0])
                    ref_onebest.append(target.target[j])
                target_back_onebest = torch.cat(target_back_onebest).view(-1, 1)
                output_back_onebest = torch.cat(output_back_onebest).view(-1, 1)
                loss_onebest = self.criterion(
                    output_back_onebest, target_back_onebest) / count_onebest

            # update the network as a combination of losses
            if update:
                total_loss = loss_onebest if self.opt.onebest else loss
                total_loss.backward()
                clip_grad_norm(self.model.parameters(), self.opt.clip)
                self.optimizer.step()

            if self.opt.onebest:
                all_loss, all_count, all_pred, all_ref = self.xent_onebest(
                    output.data.view(-1), target.indices, target.ref)
                assert all_count == count_onebest, \
                       "inconsistent count on onebest"
            else:
                all_loss, all_count, all_pred, all_ref = self.xent(
                    output.data.view(-1), lattice.ignore, target.target)

            results[index] = [(loss.data[0]*count, count),
                              (loss_onebest.data[0]*count_onebest, count_onebest),
                              (all_pred, all_ref)]

    def train(self, train_loader, epoch, val_loss):
        """Training mode."""
        if self.opt.LRDecay in ['anneal', 'stepwise']:
            self.scheduler.step()
        elif self.opt.LRDecay == 'newbob':
            self.scheduler.step(val_loss)
        self.model.train()
        avg_loss, total_count = 0, 0
        avg_loss_onebest, total_count_onebest = 0, 0
        wrapper = tqdm(train_loader, dynamic_ncols=True)
        # Looping through batches
        for lattices, targets in wrapper:
            assert len(lattices) == len(targets), \
                   "Data and targets with different lengths."
            batch_loss, batch_count = 0, 0
            batch_loss_onebest, batch_count_onebest = 0, 0

            # CPU Hogwild training
            # Each process is one training sample in a mini-batch
            processes = []
            manager = mp.Manager()
            results = manager.list([None] * len(lattices))
            # Fork processes
            for j, (lattice, target) in enumerate(zip(lattices, targets)):
                fork = mp.Process(target=self.forward_one_lattice,
                                  args=(lattice, target, j, results, True))
                fork.start()
                processes.append(fork)
            # Wait until all processes are finished
            for fork in processes:
                fork.join()
            # Collect loss stats
            for result in results:
                batch_loss += result[0][0]
                batch_count += result[0][1]
                batch_loss_onebest += result[1][0]
                batch_count_onebest += result[1][1]
            # Compute average losses and increment counters
            avg_loss = Trainer.moving_average(
                avg_loss, total_count, batch_loss, batch_count)
            avg_loss_onebest = Trainer.moving_average(
                avg_loss_onebest, total_count_onebest,
                batch_loss_onebest, batch_count_onebest)
            total_count += batch_count
            total_count_onebest += batch_count_onebest
            learning_rate = self.optimizer.param_groups[0]['lr']
            # Set tqdm display elements
            wrapper.set_description("".ljust(7) + 'Train')
            postfix = OrderedDict()
            postfix['allarc'] = '%.4f' %Trainer.mean(batch_loss, batch_count)
            postfix['allarcAvg'] = '%.4f' %avg_loss
            postfix['onebest'] = '%.4f' %Trainer.mean(
                batch_loss_onebest, batch_count_onebest)
            postfix['onebestAvg'] = '%.4f' %avg_loss_onebest
            postfix['lr'] = '%.5f' %learning_rate
            wrapper.set_postfix(ordered_dict=postfix)

        self.optim_state['epoch'] = epoch - 1
        self.logger['train'].write('%d %f %f\n' %(epoch, avg_loss, avg_loss_onebest))
        print("".ljust(7) + "Training loss".ljust(16)
              + utils.color_msg('%.4f' %(avg_loss_onebest if self.opt.onebest \
                                         else avg_loss)))
        return avg_loss_onebest if self.opt.onebest else avg_loss

    def val(self, val_loader, epoch):
        """Validation mode."""
        self.model.eval()
        avg_loss, total_count = 0, 0
        avg_loss_onebest, total_count_onebest = 0, 0
        wrapper = tqdm(val_loader, dynamic_ncols=True)
        # Looping though batches
        for lattices, targets in wrapper:
            assert len(lattices) == len(targets), \
                   "Data and targets with different lengths."
            batch_loss, batch_count = 0, 0
            batch_loss_onebest, batch_count_onebest = 0, 0

            processes = []
            manager = mp.Manager()
            results = manager.list([None] * len(lattices))
            # Fork processes
            for j, (lattice, target) in enumerate(zip(lattices, targets)):
                fork = mp.Process(target=self.forward_one_lattice,
                                  args=(lattice, target, j, results, False))
                fork.start()
                processes.append(fork)
            # Wait until all processes are finished
            for fork in processes:
                fork.join()
            # Collect loss stats
            for result in results:
                batch_loss += result[0][0]
                batch_count += result[0][1]
                batch_loss_onebest += result[1][0]
                batch_count_onebest += result[1][1]
            # Compute average losses and increment counters
            avg_loss = Trainer.moving_average(
                avg_loss, total_count, batch_loss, batch_count)
            avg_loss_onebest = Trainer.moving_average(
                avg_loss_onebest, total_count_onebest,
                batch_loss_onebest, batch_count_onebest)
            total_count += batch_count
            total_count_onebest += batch_count_onebest
            wrapper.set_description("".ljust(7) + 'val'.ljust(5))
            postfix = OrderedDict()
            postfix['allarc'] = '%.4f' %Trainer.mean(batch_loss, batch_count)
            postfix['allarcAvg'] = '%.4f' %avg_loss
            postfix['onebest'] = '%.4f' %Trainer.mean(
                batch_loss_onebest, batch_count_onebest)
            postfix['onebestAvg'] = '%.4f' %avg_loss_onebest
            wrapper.set_postfix(ordered_dict=postfix)

        self.logger['val'].write('%d %f %f\n' %(epoch, avg_loss, avg_loss_onebest))
        print("".ljust(7) + "Validation loss".ljust(16)
              + utils.color_msg('%.4f' %(avg_loss_onebest if self.opt.onebest \
                                         else avg_loss)))
        return avg_loss_onebest if self.opt.onebest else avg_loss

    def test(self, val_loader, epoch):
        """Testing mode."""
        self.model.eval()
        # import pdb; pdb.set_trace()
        prediction = []
        reference = []
        posteriors = []
        avg_loss, total_count = 0, 0
        avg_loss_onebest, total_count_onebest = 0, 0
        wrapper = tqdm(val_loader, dynamic_ncols=True)
        for lattices, targets in wrapper:
            assert len(lattices) == len(targets), \
                   "Data and targets with different lengths."
            batch_loss, batch_count = 0, 0
            batch_loss_onebest, batch_count_onebest = 0, 0
            processes = []
            manager = mp.Manager()
            results = manager.list([None] * len(lattices))
            # Fork processes
            for j, (lattice, target) in enumerate(zip(lattices, targets)):
                fork = mp.Process(target=self.forward_one_lattice,
                                  args=(lattice, target, j, results, False))
                fork.start()
                processes.append(fork)
            # Wait until all processes are finished
            for fork in processes:
                fork.join()
            # Colelct loss stats
            for result in results:
                batch_loss += result[0][0]
                batch_count += result[0][1]
                batch_loss_onebest += result[1][0]
                batch_count_onebest += result[1][1]
                prediction += result[2][0]
                reference += result[2][1]
            # Compute average losses and increment counters
            avg_loss = Trainer.moving_average(
                avg_loss, total_count, batch_loss, batch_count)
            avg_loss_onebest = Trainer.moving_average(
                avg_loss_onebest, total_count_onebest,
                batch_loss_onebest, batch_count_onebest)
            total_count += batch_count
            total_count_onebest += batch_count_onebest
            wrapper.set_description("".ljust(7) + 'Test epoch %i' %epoch)
            postfix = OrderedDict()
            postfix['allarc'] = '%.4f' %Trainer.mean(batch_loss, batch_count)
            postfix['allarcAvg'] = '%.4f' %avg_loss
            postfix['onebest'] = '%.4f' %Trainer.mean(
                batch_loss_onebest, batch_count_onebest)
            postfix['onebestAvg'] = '%.4f' %avg_loss_onebest
            wrapper.set_postfix(ordered_dict=postfix)

            for lattice, target in zip(lattices, targets):
                for i, edge_data in enumerate(lattice.edges):
                    if self.opt.onebest:
                        if i in target.indices:
                            posteriors.append(edge_data[-1])
                    else:
                        if i not in lattice.ignore:
                            posteriors.append(edge_data[-1])
            assert len(posteriors) == len(prediction), "wrong lengths"

        self.logger['test'].write('%f %f\n' %(avg_loss, avg_loss_onebest))
        print("".ljust(7) + "Test loss".ljust(16)
              + utils.color_msg('%.4f' %(avg_loss_onebest if self.opt.onebest \
                                         else avg_loss)))

        prediction = np.array(prediction)
        reference = np.array(reference)
        posteriors = np.array(posteriors)
        if self.opt.onebest:
            return avg_loss_onebest, prediction, reference, posteriors
        else:
            return avg_loss, prediction, reference, posteriors

def create_trainer(model, criterion, opt, optim_state):
    """New Trainer object."""
    trainer = Trainer(model, criterion, opt, optim_state)
    return trainer
