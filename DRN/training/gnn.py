"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import time 
import math

# Externals
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import DataParallel
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
import tqdm

import numpy as np

from models import get_model, get_losses
# Locals
from .base import base
from .Try_Optimizers import CyclicLRWithRestarts
from training.semiparam import get_dscb_loss_l2, get_dscb_loss_sigmoid, get_dscb_loss_sigmoid_minalpha

class GNNTrainer(base):
    """Trainer code for basic classification problems with binomial cross entropy."""

    def __init__(self, real_weight=1, fake_weight=1, category_weights=None, parallel=False, acc_rate=1, **kwargs):
        self.acc_rate = acc_rate
        super(GNNTrainer, self).__init__(**kwargs)
        if category_weights is None:
            self._category_weights = torch.tensor([fake_weight, real_weight]).to(self.device)
        else:
            self._category_weights = torch.tensor(category_weights.astype(np.float32)).to(self.device)

        self.parallel = parallel

    def build_model(self, name='EdgeNet', loss_func='binary_cross_entropy',
                    optimizer='Adam', lr_sched = 'Cyclic',
                    min_lr=1e-7, max_lr=1e-3, restart_period=100, 
                    gamma=1.0,
                    batch_size=2000, epoch_size=10000, warm = None,
                    thresh = None, reg = None, epsilon = None, minalpha = None,
                    **model_args):
        """Instantiate our model"""

        # Construct the model
        self.model = get_model(name=name, actually_jit=True, **model_args).to(self.device)
        if self.parallel:
            self.model = DataParallel(self.model,device_ids=list(range(torch.cuda.device_count()))).to(self.device)
            print("NUMBER OF CUDA CORES:",torch.cuda.device_count())

        if warm is not None:
            print("Warm starting with parameters from checkpoint",warm)
            state = torch.load(warm, map_location = self.device)['model']
            modelstate = self.model.state_dict()
            newstate = {}
            keys = list(state.keys())  
            modelkeys = modelstate.keys()
            if not keys[0].startswith('module.drn.'):
                for key in keys:
                    if 'edgeconv' in key:
                        splits = key.split('.')
                        rest = '.'.join(splits[1:])
                        index = int(splits[0][8:]) - 1
                        newstate[f'drn.agg_layers.{index}.{rest}'] = state[key]
                    else:
                        newstate['drn.'+key] = state[key]
            else:
                newstate = state

            print("Oldkeys",modelstate.keys())
            print("Newkeys",newstate.keys())
            
            for key in modelkeys:
                if newstate[key].shape != modelstate[key].shape:
                    print("implimenting partial match in %s due to shape mismatch"%key)
                    #del newstate[key]
                    tmp = newstate[key]
                    works = modelstate[key]
                    ndim = len(tmp.shape)
                    if ndim==1:
                        #print("TMP",tmp.shape)
                        #print(tmp)
                        #print("WORKS",works.shape)
                        #print(works)
                        length = tmp.shape[0]
                        works[:length] = tmp
                        #print("newWORKS",works.shape)
                        #print(works)
                    elif ndim==2:
                        #print()
                        #print("TMP")
                        #print(tmp)
                        #print("WORKS")
                        #print(works)
                        length = tmp.shape[0]
                        width = tmp.shape[1]
                        works[:length,:width] = tmp
                        #print("newWORKS")
                        #print(works)
                    newstate[key] = works

            self.model.load_state_dict(newstate,strict=True)

        # Construct the loss function
        get_losses()
        if loss_func == 'dscb_loss_l2':
            self.loss_func = get_dscb_loss_l2(thresh, reg)
        elif loss_func == 'dscb_loss_sigmoid':
            self.loss_func = get_dscb_loss_sigmoid(thresh, epsilon)
        elif loss_func == 'dscb_loss_sigmoid_minalpha':
            self.loss_func = get_dscb_loss_sigmoid_minalpha(thresh, epsilon, minalpha)
        else:
            self.loss_func = getattr(nn.functional, loss_func)

        print(self.loss_func)

        # Construct the optimizer
        self.optimizer = getattr(torch.optim, optimizer)(
            self.model.parameters(), lr=max_lr)

        if lr_sched == 'Cyclic':
            self.lr_scheduler = CyclicLRWithRestarts(self.optimizer, 
                batch_size, epoch_size, restart_period=restart_period, 
                min_lr = min_lr, policy = 'cosine')
        elif lr_sched == 'Const':
            self.lr_scheduler = LambdaLR(self.optimizer, lambda epoch:1)
        elif lr_sched == 'TorchCyclic':
            self.lr_scheduler = CyclicLR(self.optimizer, min_lr, max_lr, int(np.ceil(epoch_size/batch_size)*restart_period), cycle_momentum=False, scale_mode='cycle', scale_fn = lambda i:gamma**(i))
        else:
            print("Invalid learning rate schedule!")

        self.lr_sched = lr_sched

    # @profile
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        if '{}'.format(self.device) != 'cpu':
            torch.cuda.reset_max_memory_allocated(self.device)
        self.model.train()        
        summary = dict()
        summary['acc_loss']=[]
        summary['acc_lr']=[]
        sum_loss = 0.
        start_time = time.time()
        # Loop over training batches
        total = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        t = tqdm.tqdm(enumerate(data_loader),total=int(math.ceil(total/batch_size)))
        cat_weights = self._category_weights
        
        acc_rate = self.acc_rate
        acc_norm = 1./acc_rate        
        acc_loss = 0.
        
        self.optimizer.zero_grad()
        for i,data in t:            
            data = data.to(self.device)
            batch_target = data.y
            if self.loss_func == F.binary_cross_entropy:
                #binary cross entropy expects a weight for each event in a batch
                #categorical cross entropy ex
                batch_weights_real = batch_target*self._category_weights[1]
                batch_weights_fake = (1 - batch_target)*self._category_weights[0]
                cat_weights = batch_weights_real + batch_weights_fake            
            if self.parallel:
                batch_output = self.model(data.to_data_list())
            else:
                batch_output = self.model(data)

            batch_loss = acc_norm * self.loss_func(batch_output,batch_target) 
            batch_loss.backward()
            batch_loss_item = batch_loss.item()
            acc_loss += batch_loss_item
            sum_loss += batch_loss_item
            if(acc_rate ==1 or (i+1) % acc_rate == 0 or (i+1) == total): 
                self.optimizer.step()
                self.optimizer.zero_grad()
                t.set_description("loss = %.5f" % acc_loss )
                summary['acc_loss'].append(acc_loss)
                summary['acc_lr'].append(self.optimizer.param_groups[0]['lr'])
                if self.lr_sched == 'TorchCyclic':
                    self.lr_scheduler.step()
                acc_loss = 0.
                      
            #self.logger.debug('  batch %i, loss %f', i, batch_loss.item())


        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = acc_rate * sum_loss / (i + 1)
        self.logger.debug(' Processed %i batches', (i + 1))
        self.logger.info('  Training loss: %.5f', summary['train_loss'])
        self.logger.info('  Learning rate: %.5f', summary['lr'])
        
        if '{}'.format(self.device) != 'cpu':
            self.logger.info('  Max memory usage: %d', torch.cuda.max_memory_allocated(self.device))
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.zero_grad()
        #torch.cuda.empty_cache()
        self.model.eval()
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        start_time = time.time()
        # Loop over batches
        total = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        t = tqdm.tqdm(enumerate(data_loader),total=int(math.ceil(total/batch_size)))
        num = torch.zeros_like(self._category_weights)
        denm = torch.zeros_like(self._category_weights)
        
        cat_wgt_shape = self._category_weights.shape[0]
      
        confusion_num = torch.zeros([cat_wgt_shape,cat_wgt_shape]).to(self.device)
        confusion_denm = torch.zeros([cat_wgt_shape,cat_wgt_shape]).to(self.device)
        
        for i, data in t:            
            # self.logger.debug(' batch %i', i)
            batch_input = data.to(self.device)
            batch_target = data.y
            if self.parallel:
                batch_output = self.model(batch_input.to_data_list())
            else:
                batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            sum_loss += batch_loss.item()
            # Count number of correct predictions
            #print(batch_output)
            #print('torch.max',torch.argmax(batch_output,dim=-1))
            
            truth_cat_counts = torch.unique(batch_target, return_counts = True)
            pred = torch.argmax(batch_output,dim=-1)
            
            for j in range(cat_wgt_shape):
                 pass
#                cat_counts = torch.unique(pred[batch_target == j], return_counts=True)                
#                confusion_num[:,j][cat_counts[0]] += cat_counts[1].float()
#                confusion_denm[j,:][truth_cat_counts[0]] += truth_cat_counts[1].float()
                        
#            matches = (pred == batch_target)
#            trues_by_cat = torch.unique(pred[matches], return_counts=True)
            

#            num[trues_by_cat[0]] += trues_by_cat[1].float()
#            denm[truth_cat_counts[0]] += truth_cat_counts[1].float()
                        
            
#            sum_correct += matches.sum().item()
#            sum_total += matches.numel()
            #self.logger.debug(' batch %i loss %.3f correct %i total %i',
            #                  i, batch_loss.item(), matches.sum().item(),
            #                  matches.numel())
        #torch.cuda.empty_cache()
        #self.logger.debug('loss %.5f cat effs %s',torch.true_divide(sum_loss, (i + 1)), np.array_str((num/denm).cpu().numpy()))
        #self.logger.debug('loss %.5f cat confusions:\n %s',
        #                  torch.true_divide(sum_loss , (i + 1)),
        #                  np.array_str((confusion_num/confusion_denm).cpu().numpy()))
        #self.logger.debug('loss %.5f cat true counts %s',torch.true_divide(sum_loss , (i + 1)), (denm).cpu().numpy())
        #self.logger.debug('loss %.5f cat wgt counts %s',torch.true_divide(sum_loss , (i + 1)), (self._category_weights*denm).cpu().numpy())
        if self.lr_sched != "TorchCyclic":
            self.lr_scheduler.step()
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = torch.true_divide(sum_loss , (i + 1))
#        summary['valid_acc'] = sum_correct / sum_total
        summary['valid_acc'] = 0
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)
        self.logger.info('  Validation loss: %.5f acc: %.5f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary


def _test():
    t = GNNTrainer(output_dir='./')
    t.build_model()
