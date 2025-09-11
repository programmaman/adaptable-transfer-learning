import time
import warnings
import os

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv

warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans


class NegativeSampler(object):
    """
    Negative Sampler for Graph Neural Networks
    Based on the official GPPT repository implementation
    """

    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n * self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type='gcn'):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregator_type))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator_type))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggregator_type))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, aggregator_type))
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def get_e(self):
        return self.embedding_x

    def get_pre(self):
        return self.pre

    def forward(self, blocks, x):
        h = self.dropout(x)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        self.embedding_x = h
        self.pre = self.fc(h)
        return h

    def forward_smc(self, g, x):
        h = self.dropout(x)
        for layer_idx, layer in enumerate(self.layers):
            h = layer(g, h)
            if layer_idx != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        self.embedding_x = h
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()).to(g.device),
                sampler,
                device=device if num_workers == 0 else None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in dataloader:  # tqdm.tqdm(dataloader):
                block = blocks[0]
                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()
                # gc.collect()
                # torch.cuda.empty_cache()

            x = y
        return y


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 center_num):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_classes = n_classes
        self.center_num = center_num
        self.n_hidden = n_hidden
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))

        self.prompt = nn.Linear(2 * n_hidden, self.center_num, bias=False)

        self.pp = nn.ModuleList()
        for i in range(self.center_num):
            self.pp.append(nn.Linear(2 * n_hidden, n_classes, bias=False))

    def model_to_array(self, args):
        """Convert model state dict to array for loading pre-trained weights"""
        model_path = f'./logs/pretrained/{args.dataset}_model_{args.file_id}.pt'
        if not os.path.exists(model_path):
            print(f"Pre-trained model not found at {model_path}")
            return None

        s_dict = torch.load(model_path, map_location=self.device)
        keys = list(s_dict.keys())
        res = s_dict[keys[0]].view(-1)
        for i in range(1, len(keys)):
            res = torch.cat((res, s_dict[keys[i]].view(-1)))
        return res

    def array_to_model(self, args):
        """Load pre-trained weights from array format"""
        arr = self.model_to_array(args)
        if arr is None:
            return

        model_path = f'./logs/pretrained/{args.dataset}_model_{args.file_id}.pt'
        m_m = torch.load(model_path, map_location=self.device)
        indice = 0
        s_dict = self.state_dict()
        for name, param in m_m.items():
            if name in s_dict:  # Only load compatible layers
                length = torch.prod(torch.tensor(param.shape))
                s_dict[name] = arr[indice:indice + length].view(param.shape)
                indice = indice + length
        self.load_state_dict(s_dict, strict=False)

    def load_parameters(self, args):
        """Load pre-trained parameters"""
        self.args = args
        self.array_to_model(args)

    def weigth_init(self, graph, inputs, label, index):
        h = self.dropout(inputs)
        for layer_idx, layer in enumerate(self.layers):
            h = layer(graph, h)
            if layer_idx != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        h = self.activation(h)
        graph.ndata['h'] = h
        graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neighbor'))
        neighbor = graph.ndata['neighbor']
        h = torch.cat((h, neighbor), dim=1)

        features = h[index]
        labels = label[index.long()]
        cluster = KMeans(n_clusters=self.center_num, random_state=0).fit(
            features.detach().cpu())

        temp = torch.FloatTensor(cluster.cluster_centers_).to(self.device)
        self.prompt.weight.data.copy_(temp)

        p = []
        for i in range(self.n_classes):
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                p.append(features[class_mask].mean(dim=0).view(1, -1))
            else:
                # Handle empty classes by using random initialization
                p.append(torch.randn(1, features.shape[1]).to(features.device))

        temp = torch.cat(p, dim=0)
        for i in range(self.center_num):
            self.pp[i].weight.data.copy_(temp)

    def update_prompt_weight(self, h):
        cluster = KMeans(n_clusters=self.center_num, random_state=0).fit(
            h.detach().cpu())
        temp = torch.FloatTensor(cluster.cluster_centers_).to(self.device)
        self.prompt.weight.data.copy_(temp)

    def get_mul_prompt(self):
        pros = []
        for name, param in self.named_parameters():
            if name.startswith('pp.'):
                pros.append(param)
        return pros

    def get_prompt(self):
        for name, param in self.named_parameters():
            if name.startswith('prompt.weight'):
                pro = param
        return pro

    def get_mid_h(self):
        return self.fea

    def forward(self, graph, inputs):
        h = self.dropout(inputs) if self.dropout else inputs
        for layer_idx, layer in enumerate(self.layers):
            h_dst = h[:graph[layer_idx].num_dst_nodes()]  # <---
            h = layer(graph[layer_idx], (h, h_dst))
            if layer_idx != len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout:
                    h = self.dropout(h)
        h = self.activation(h)
        h_dst = self.activation(h_dst)
        neighbor = h_dst
        h = torch.cat((h, neighbor), dim=1)
        self.fea = h

        out = self.prompt(h)
        index = torch.argmax(out, dim=1)
        out = torch.zeros(h.shape[0], self.n_classes).to(self.device)
        for i in range(self.center_num):
            mask = (index == i)
            if mask.sum() > 0:
                out[mask] = self.pp[i](h[mask])
        return out

    def forward_smc(self, graph, inputs):
        """Single graph forward pass for evaluation"""
        h = self.dropout(inputs) if self.dropout else inputs
        for layer_idx, layer in enumerate(self.layers):
            h = layer(graph, h)
            if layer_idx != len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout:
                    h = self.dropout(h)
        h = self.activation(h)

        # Compute neighbor features for prompt tuning
        graph.ndata['h'] = h
        graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neighbor'))
        neighbor = graph.ndata['neighbor']
        h = torch.cat((h, neighbor), dim=1)
        self.fea = h

        out = self.prompt(h)
        index = torch.argmax(out, dim=1)
        out = torch.zeros(h.shape[0], self.n_classes).to(self.device)
        for i in range(self.center_num):
            mask = (index == i)
            if mask.sum() > 0:
                out[mask] = self.pp[i](h[mask])
        return out


def main(args):
    # This is the original main function that requires utils
    # For testing purposes, we'll use the run_gppt_experiment.py script instead
    pass
