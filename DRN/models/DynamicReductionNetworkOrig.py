import os
import os.path as osp
import math

import numpy as np
import torch
import gc
import torch.nn as nn
from torch.nn.functional import softplus
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph

from torch_geometric.nn import EdgeConv, NNConv
from torch_geometric.utils import normalized_cut
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.nn import (graclus, 
                                max_pool, max_pool_x, global_max_pool,
                                avg_pool, avg_pool_x, global_mean_pool, 
                                global_add_pool)

transform = T.Cartesian(cat=False)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

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
    @param output_dim: dimensio of output
    
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
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1, k=16, aggr='add', norm=None, 
            loop=True, pool='max',
            agg_layers=2, mp_layers=2, in_layers=1, out_layers=3,
            graph_features = False,
            latent_probe=None):
        super(DynamicReductionNetwork, self).__init__()

        self.graph_features = graph_features

        if latent_probe is not None and (latent_probe>agg_layers+1 or latent_probe<-1*agg_layers-1):
            print("Error: asked for invalid latent_probe layer")
            return
        
        if latent_probe is not None and latent_probe < 0:
            latent_probe = agg_layers+1 - latent_probe

        if latent_probe is not None:
            print("Probing latent features after %dth layer"%latent_probe)

        self.latent_probe = latent_probe

        self.loop = loop
        self.agg_layers = agg_layers

        print("Pooling with",pool)
        print("Using self-loops" if self.loop else "Not using self-loops")
        print("There are",self.agg_layers,'aggregation layers')

        if norm is None:
            norm = torch.ones(input_dim)

        #normalization vector
        self.datanorm = nn.Parameter(norm)
        
        self.k = k

        #construct inputnet
        in_layers_l = []
        in_layers_l += [nn.Linear(input_dim, hidden_dim),
                nn.ELU()]

        for i in range(in_layers-1):
            in_layers_l += [nn.Linear(hidden_dim, hidden_dim), 
                    nn.ELU()]

        self.inputnet = nn.Sequential(*in_layers_l)


        #construct aggregation layers
        for i in range(agg_layers):
            #construct message passing network
            mp_layers_l = []

            for j in range(mp_layers-1):
                mp_layers_l += [nn.Linear(2*hidden_dim, 2*hidden_dim),
                        nn.ELU()]

            mp_layers_l += [nn.Linear(2*hidden_dim, hidden_dim),
                    nn.ELU()]
           
            convnn = nn.Sequential(*mp_layers_l)
            
            name = 'edgeconv%d'%(i+1)
            setattr(self,name,EdgeConv(nn=convnn, aggr=aggr))

        #construct outputnet
        out_layers_l = []

        for i in range(out_layers-1):
            out_layers_l += [nn.Linear(hidden_dim, hidden_dim), 
                    nn.ELU()]

        out_layers_l += [nn.Linear(hidden_dim, output_dim)]

        self.output = nn.Sequential(*out_layers_l)


        #use appropriate pooling method
        if pool == 'max':
            self.poolfunc = max_pool
            self.x_poolfunc = max_pool_x
            self.global_poolfunc = global_max_pool
        elif pool == 'mean':
            self.poolfunc = avg_pool
            self.x_poolfunc = avg_pool_x
            self.global_poolfunc = global_mean_pool
        elif pool == 'add':
            self.poolfunc = avg_pool
            self.x_poolfunc = avg_pool_x
            self.global_poolfunc = global_add_pool
        else:
            print("ERROR: INVALID POOLING")

    def doLayer(self, data, i):
        ''' 
        do one aggregation layer
        @param data: current batch object
        @param i: the index of the layer to be done

        @returns: the transformed batch object. 
            if this is the last layer, instead returns (data.x, data.batch)
        '''
        name = 'edgeconv%d'%(i+1)
        edgeconv = getattr(self, name)
    
        knn = knn_graph(data.x, self.k, data.batch, loop=self.loop, flow=edgeconv.flow)
        data.edge_index = to_undirected(knn)
        data.x = edgeconv(data.x, data.edge_index)

        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))

        if i != self.agg_layers-1:
            data.edge_attr = None
            return self.poolfunc(cluster, data)
             
        else:
            return self.x_poolfunc(cluster, data.x, data.batch)

    def forward(self, data):        
        '''
        Push the batch 'data' through the network
        '''
        data.x = self.datanorm * data.x
        data.x = self.inputnet(data.x)

        if self.graph_features:
            graph_x = data.graph_x
        
        for i in range(self.agg_layers):
            if self.latent_probe is not None and i==self.latent_probe:
                return data.x
            data = self.doLayer(data, i)

        if self.agg_layers==0: #if there are no layers, format data appropriately 
            data = data.x, data.batch

        if self.latent_probe is not None and self.latent_probe == self.agg_layers:
            return data[0]

        x = self.global_poolfunc(*data)

        if self.latent_probe is not None and self.latent_probe == self.agg_layers+1:
            return x

        if self.graph_features:
            x = torch.cat((x, graph_x), 1)

        x = self.output(x).squeeze(-1)

        return x
