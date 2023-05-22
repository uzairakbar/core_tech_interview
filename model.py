import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from collections import defaultdict



class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()
    
    def forward(self, G, h_dict):
        return G, {k : F.leaky_relu(h) for k, h in h_dict.items()}


class Interpolate(nn.Module):
    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.size = size
    
    def forward(self, G, h_dict):
        return G, {
            k : F.interpolate(h[:, None], size=self.size)[:, 0]
                for k, h in h_dict.items()
        }


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_sizes, out_sizes, cetypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                ','.join(cetype) : nn.Linear(
                        in_sizes[cetype[0]], out_sizes[cetype[-1]]
                    ) 
                    for cetype in cetypes
        })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for cetype in G.canonical_etypes:
            srctype, etype, dsttype = cetype
            # Compute W_r * h
            Wh = self.weight[','.join(cetype)](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s_%s_%s' % cetype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[cetype] = (
                fn.copy_u('Wh_%s_%s_%s' % cetype, 'm'),
                fn.mean('m', 'h')
            )
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return G, {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, num_hidden):
        super(HeteroRGCN, self).__init__()
        
        in_sizes = defaultdict(lambda: in_size)
        hidden_sizes = defaultdict(lambda: hidden_size)
        out_sizes = defaultdict(lambda: out_size)
        for ntype in G.ntypes:
            if "h" in G.nodes[ntype].data:
                h_size = G.nodes[ntype].data["h"].shape[1]
                in_sizes[ntype] = h_size
                hidden_sizes[ntype] = h_size

        # Use trainable node embeddings as featureless inputs.
        embed_dict = {
            ntype : nn.Parameter(
                        G.nodes[ntype].data["h"],
                        requires_grad=False
                        )
                        if "h" in G.nodes[ntype].data else
                    nn.Parameter(
                        torch.Tensor(
                            G.number_of_nodes(ntype), in_sizes[ntype]
                        ),
                        requires_grad=True
                        )
                        for ntype in G.ntypes
        }
        
        for key, embed in embed_dict.items():
            if embed.requires_grad:
                nn.init.xavier_uniform_(embed)
        
        self.embed = nn.ParameterDict(embed_dict)

        # create layers
        self.layers = mySequential()
        for i in range(num_hidden):
            self.layers.add_module(
                f"layer{i}",
                HeteroRGCNLayer(
                hidden_sizes if i else in_sizes,
                hidden_sizes,
                G.canonical_etypes
                )
            )
            self.layers.add_module(f"relu{i}", Relu())
        self.layers.add_module(
            f"layer{num_hidden}",
            HeteroRGCNLayer(
            hidden_sizes,
            out_sizes,
            G.canonical_etypes
            )
        )

    def forward(self, G):
        _, h_dict = self.layers(G, self.embed)

        # get paper logits
        return h_dict['paper']