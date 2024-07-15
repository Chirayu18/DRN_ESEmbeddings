import torch
import torch.nn as nn
from .DynamicReductionNetworkJit import DynamicReductionNetworkJit
from .DynamicReductionNetworkOld import DynamicReductionNetworkOld

class DynamicReductionNetwork(nn.Module):
    '''
    This model iteratively contracts nearest neighbour graphs 
    until there is one output node.
    The latent space trained to group useful features at each level
    of aggregration.
    This allows single quantities to be regressed from complex point counts
    in a location and orientation invariant way.
    One encoding layer is used to abstract away the input features.

    @param input_dim: dimension of input features
    @param hidden_dim: dimension of hidden layers
    @param output_dim: dimension of output
    
    @param k: size of k-nearest neighbor graphs
    @param aggr: message passing aggregation scheme. 
    @param norm: feature normaliztion. None is equivalent to all 1s (ie no scaling)
    @param loop: boolean for presence/absence of self loops in k-nearest neighbor graphs
    @param pool: type of pooling in aggregation layers. Choices are 'add', 'max', 'mean'
    
    @param agg_layers: number of aggregation layers. Must be >=0
    @param mp_layers: number of layers in message passing networks. Must be >=1
    @param in_layers: number of layers in inputnet. Must be >=1
    @param out_layers: number of layers in outputnet. Must be >=1
    '''
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, k=16, aggr='add', norm=None, 
                 loop=True, pool='max',
                 agg_layers=2, mp_layers=2, in_layers=1, out_layers=3,
                 graph_features = False,
                 latent_probe=None,
                 actually_jit=True,
                 original_drn=False,
    ):
        super(DynamicReductionNetwork, self).__init__()
        DRN = DynamicReductionNetworkJit
        if original_drn:
            DRN = DynamicReductionNetworkOld
        
        drn = DRN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            k=k,
            aggr=aggr,
            norm=norm,
            agg_layers=agg_layers,
            mp_layers=mp_layers,
                in_layers=in_layers,
            out_layers=out_layers,
            graph_features=graph_features,
            latent_probe=latent_probe
        )
        if not original_drn:
            if actually_jit:
                self.drn = torch.jit.script(drn)
            else:
                self.drn = drn
        else:
            self.drn = drn

    def forward(self, data):
        '''
        Push the batch 'data' through the network
        '''
        if isinstance(self.drn, DynamicReductionNetworkOld):
            return self.drn(data)
        #print(torch.zeros((data.xECAL.shape[0]),dtype=torch.int64,device=data.xECAL.device))
        #print(type(data))
        
        return self.drn(
            data.xECAL,
            data.xES,
            data.batch if hasattr(data, 'batch') else torch.zeros((data.xECAL.shape[0], ),
                                                                  dtype=torch.int64,
                                                                  device=data.xECAL.device),
            data.graph_x if hasattr(data, 'graph_x') else None
        )
