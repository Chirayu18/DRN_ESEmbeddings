"""
Common PyTorch trainer code.
"""

import logging
# System
import os

# Externals
import numpy as np
import torch


class base(object):
    """
    Base class for PyTorch trainers.
    This implements the common training logic,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, device='cpu', distributed=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir)
                           if output_dir is not None else None)
        self.device = device
        self.distributed = distributed
        self.summaries = {}

    def print_model_summary(self):
        """Override as needed"""
        self.logger.info(
            'Model: \n%s\nParameters: %i' %
            (self.model, sum(p.numel()
                             for p in self.model.parameters()))
        )

    @staticmethod
    def get_model_fname(model):
        import hashlib
        model_name = type(model).__name__
        model_params = sum(p.numel() for p in model.parameters())        
        model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
        model_user = os.environ['USER']
        model_fname = '%s_%d_%s_%s'%(model_name, model_params,
                                 model_cfghash, model_user)
        return model_fname
        
    def save_summary(self, summaries):
        """Save summary information"""
        for (key, val) in summaries.items():
            summary_vals = self.summaries.get(key, [])
            self.summaries[key] = summary_vals + [val]

    def write_summaries(self):
        assert self.output_dir is not None
        summary_file = os.path.join(self.output_dir, 'summaries.npz')
        self.logger.info('Saving summaries to %s' % summary_file)
        np.savez(summary_file, **self.summaries)

    def write_checkpoint(self, checkpoint_id, epoch, loss, best=False):
        """Write a checkpoint for the model"""
        assert self.output_dir is not None
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        fname = self.get_model_fname(self.model)
        checkpoint_file = ''
        if best:
            checkpoint_file = 'model_checkpoint_%s.best.pth.tar' % ( fname )
        else:
            checkpoint_file = 'model_checkpoint_%s_%03i.pth.tar' % ( fname, checkpoint_id )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkdict = {
                'epoch' :  epoch,
                'model' :  self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'loss' : loss,
                }
        torch.save(checkdict, os.path.join(checkpoint_dir, checkpoint_file))

    def build_model(self):
        """Virtual method to construct the model(s)"""
        raise NotImplementedError

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader, extra_output=None):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, valid_data_loader=None, early_stopping_rounds=0, resume=False):
        """Run the model training"""

        rounds_wihout_improvement = 0
        # Loop over epochs

        best_valid_loss = 99999
        for i in range(n_epochs):
            self.logger.info('Epoch %i' % i)
            summary = dict(epoch=i)            
            # Train on this epoch
            sum_train = self.train_epoch(train_data_loader)            
            summary.update(sum_train)
            # Evaluate on this epoch
            sum_valid = None
            if valid_data_loader is not None:
                sum_valid = self.evaluate(valid_data_loader)
                summary.update(sum_valid)
                
                if sum_valid['valid_loss'] < best_valid_loss:
                    best_valid_loss = sum_valid['valid_loss']
                    self.logger.debug('Checkpointing new best model with loss: %.5f', best_valid_loss)
                    self.write_checkpoint(checkpoint_id=i,best=True, epoch=i, loss=sum_valid['valid_loss'])                
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

            # Save summary, checkpoint
            self.save_summary(summary)
            if self.output_dir is not None:
                self.write_checkpoint(checkpoint_id=i, epoch=i, loss=sum_valid['valid_loss'])
                self.write_summaries()

            if early_stopping_rounds>0 and rounds_without_improvement>early_stopping_rounds:
                break;

        return self.summaries
