
import torch.nn as nn
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops
from torch_geometric.loader import DataLoader

from experiments.experiment_utils import generate_synthetic_graph


class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.initialize()

    def forward(self, x):
        return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x):
        return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


def transfer(args, config, gpu_id, is_reduction,
             pretrain_dataset=None, test_dataset=None):
    import math
    from torch_geometric.utils import negative_sampling

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load / prep datasets ---
    if pretrain_dataset is None or test_dataset is None:
        print("Loading datasets...")
        pretrain_datapath = os.path.join('./datasets', args.pretrain_dataset)
        test_datapath = os.path.join('./datasets', args.test_dataset)
        pretrain_dataset = get_dataset(pretrain_datapath, args.pretrain_dataset)[0]
        test_dataset = get_dataset(test_datapath, args.test_dataset)[0]
        print(f"Pretrain dataset: {args.pretrain_dataset}, Test dataset: {args.test_dataset}")

    # Optional feature reduction (ONLY once)
    if is_reduction:
        print("Applying SVD feature reduction...")
        feature_reduce = SVDFeatureReduction(out_channels=100)
        pretrain_dataset = feature_reduce(pretrain_dataset)
        test_dataset    = feature_reduce(test_dataset)

    # Ensure self-loops and push to device
    pretrain_dataset.edge_index = add_remaining_self_loops(pretrain_dataset.edge_index)[0]
    test_dataset.edge_index     = add_remaining_self_loops(test_dataset.edge_index)[0]
    pretrain_dataset = pretrain_dataset.to(device)
    test_dataset    = test_dataset.to(device)

    # Heuristic for big graphs: avoid any N×N dense tensors
    BIG_GRAPH_THRESHOLD = 5000
    is_big = int(test_dataset.num_nodes) > BIG_GRAPH_THRESHOLD
    if is_big:
        print(f"[Info] Big graph detected (N={test_dataset.num_nodes}). Using memory-safe losses.")
    else:
        print(f"[Info] Small/medium graph (N={test_dataset.num_nodes}).")

    # --- Models ---
    print("Initializing models...")
    in_dim = int(pretrain_dataset.x.shape[1])
    gnn = GNN(in_dim, config['output_dim'], act(config['activation']),
              config['gnn_type'], config['num_layers'])
    model_path = "./pre_trained_gnn/{}.{}.{}.{}.pth".format(
        args.pretrain_dataset, args.pretext, config['gnn_type'], args.is_reduction
    )
    print(f"Loading pre-trained GNN weights from {model_path}...")
    gnn.load_state_dict(torch.load(model_path, map_location='cpu'))
    gnn.to(device)
    gnn.eval()
    for p in gnn.conv.parameters():
        p.requires_grad = False

    print("Setting up GNN with LoRA...")
    gnn2 = GNNLoRA(in_dim, config['output_dim'], act(config['activation']),
                   gnn, config['gnn_type'], config['num_layers'], r=args.r).to(device)
    gnn2.train()

    print("Setting up SMMD Loss...")
    SMMD = SMMDLoss().to(device)

    projector = Projector(int(test_dataset.x.shape[1]), in_dim).to(device)
    projector.train()

    print("Setting up Logistic Regression...")
    num_classes = int(test_dataset.y.max().item()) + 1
    logreg = LogReg(config['output_dim'], num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # --- Train/val/test masks ---
    if args.test_dataset in ['PubMed', 'CiteSeer', 'Cora']:
        if args.few:
            train_mask, val_mask, test_mask = get_few_shot_mask(test_dataset, args.shot, args.test_dataset, device)
        else:
            train_mask, val_mask, test_mask = test_dataset.train_mask, test_dataset.val_mask, test_dataset.test_mask
    else:
        if args.few:
            train_mask, val_mask, test_mask = get_few_shot_mask(test_dataset, args.shot, args.test_dataset, device)
        else:
            index = np.arange(test_dataset.x.shape[0])
            np.random.shuffle(index)
            train_mask = torch.zeros(test_dataset.x.shape[0], dtype=torch.bool, device=device)
            val_mask   = torch.zeros_like(train_mask)
            test_mask  = torch.zeros_like(train_mask)
            train_mask[index[:int(len(index)*0.1)]] = True
            val_mask[index[int(len(index)*0.1):int(len(index)*0.2)]] = True
            test_mask[index[int(len(index)*0.2):]] = True

    test_dataset.train_mask = train_mask
    test_dataset.val_mask   = val_mask
    test_dataset.test_mask  = test_mask

    train_labels = test_dataset.y[train_mask]
    val_labels   = test_dataset.y[val_mask]
    test_labels  = test_dataset.y[test_mask]

    # PPR for SMMD only on small graphs
    if is_big:
        ppr_weight = None
    else:
        ppr_weight = get_ppr_weight(test_dataset)

    pretrain_graph_loader = DataLoader(pretrain_dataset.x, batch_size=128, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam([
        {"params": projector.parameters(), 'lr': args.lr1, 'weight_decay': args.wd1},
        {"params": logreg.parameters(),     'lr': args.lr2, 'weight_decay': args.wd2},
        {"params": gnn2.parameters(),       'lr': args.lr3, 'weight_decay': args.wd3}
    ])

    # --- Memory-safe contrastive loss (no N×N mask) ---
    def batched_gct_loss_labels(z1, z2, batch_size, labels, tau=0.5):
        """
        For each row i in a batch, positives are nodes with same label as i.
        Avoids any dense N×N mask by building a [B×N] equality per batch.
        """
        N = z1.size(0)
        idx_all = torch.arange(N, device=z1.device)
        num_batches = (N - 1) // batch_size + 1

        def _sim(a, b):
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
            return a @ b.t()

        losses = []
        for b in range(num_batches):
            start = b * batch_size
            end   = min((b + 1) * batch_size, N)
            if end <= start: break
            idx = idx_all[start:end]

            refl_sim    = torch.exp(_sim(z1[idx], z1) / tau)      # [B, N]
            between_sim = torch.exp(_sim(z1[idx], z2) / tau)      # [B, N]

            # positives (same label): [B, N]
            same = (labels[idx].unsqueeze(1) == labels.unsqueeze(0)).float()

            # denom = sum over all similarities except self-reflections in this block
            denom = refl_sim.sum(1) + between_sim.sum(1)
            # subtract the self terms on the local diagonal slice
            diag_slice = torch.diagonal(refl_sim[:, start:end])
            denom = denom - diag_slice

            numer = (same * between_sim).sum(1) + 1e-12
            loss_b = -torch.log(numer / (denom + 1e-12))
            losses.append(loss_b)
        return torch.cat(losses) if len(losses) else torch.zeros((), device=z1.device)

    max_acc = 0.0
    max_test_acc = 0.0
    max_epoch = 0

    # --- Training loop ---
    for epoch in range(0, args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        projector.train(); logreg.train(); gnn2.train()
        optimizer.zero_grad()

        # Project to pretrain feature space and embed
        feature_map = projector(test_dataset.x)                                  # [N, in_dim]
        emb, emb1, emb2 = gnn2(feature_map, test_dataset.edge_index)            # [N, d], [N,d], [N,d]

        # Losses
        if ppr_weight is None:
            smmd_loss_f = torch.tensor(0.0, device=device)  # skip SMMD for big graphs
        else:
            smmd_loss_f = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)

        # contrastive (label-based, memory safe)
        ct_loss = 0.5 * (
            batched_gct_loss_labels(emb1, emb2, 1000, test_dataset.y, args.tau).mean() +
            batched_gct_loss_labels(emb2, emb1, 1000, test_dataset.y, args.tau).mean()
        )

        # classification
        logits = logreg(emb)
        train_logits = logits[train_mask]
        cls_loss = loss_fn(train_logits, train_labels)

        # edge-based reconstruction (no dense N×N)
        pos_edge_index = test_dataset.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=test_dataset.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        log_probs = torch.softmax(logits, dim=1)
        logits_pos = (log_probs[pos_edge_index[0]] * log_probs[pos_edge_index[1]]).sum(dim=1)
        logits_neg = (log_probs[neg_edge_index[0]] * log_probs[neg_edge_index[1]]).sum(dim=1)
        target_edges = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)])
        logits_edges = torch.cat([logits_pos, logits_neg])
        loss_rec = F.binary_cross_entropy_with_logits(logits_edges, target_edges)

        # total
        loss = args.l1 * cls_loss + args.l2 * smmd_loss_f + args.l3 * ct_loss + args.l4 * loss_rec
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            projector.eval(); logreg.eval(); gnn2.eval()
            val_preds = logits[val_mask].argmax(dim=1)
            test_preds = logits[test_mask].argmax(dim=1)
            train_acc = (train_logits.argmax(dim=1) == train_labels).float().mean()
            val_acc   = (val_preds == val_labels).float().mean()
            test_acc  = (test_preds == test_labels).float().mean()
            print(f"Epoch: {epoch:03d}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.6f}, test_acc: {test_acc:.6f}")

            if val_acc > max_acc:
                max_acc = float(val_acc)
                max_test_acc = float(test_acc)
                max_epoch = epoch + 1

        # (optional) free some caches per epoch
        if torch.cuda.is_available() and (epoch % 5 == 0):
            torch.cuda.empty_cache()

    print('epoch: {}, val_acc: {:4f}, test_acc: {:4f}'.format(max_epoch, max_acc, max_test_acc))

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    # Compute metrics on the final/best model state
    with torch.no_grad():
        preds = logits[test_mask].argmax(dim=1).detach().cpu().numpy()
        y_true = test_labels.detach().cpu().numpy()

        precision = precision_score(y_true, preds, average="macro", zero_division=0)
        recall = recall_score(y_true, preds, average="macro", zero_division=0)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)

        # Optional: AUC and AP (only for binary classification)
        auc = ap = None
        if len(set(y_true)) == 2:
            probs = torch.softmax(logits[test_mask], dim=1)[:, 1].detach().cpu().numpy()
            auc = roc_auc_score(y_true, probs)
            ap = average_precision_score(y_true, probs)

    result_path = './result'
    mkdir(result_path)
    with open(os.path.join(result_path, 'GraphLoRA.txt'), 'a') as f:
        if args.few:
            f.write('Few: True, r: %d, Shot: %d, %s to %s: val_acc: %f, test_acc: %f\n' %
                    (args.r, args.shot, args.pretrain_dataset, args.test_dataset, max_acc, max_test_acc))
        else:
            f.write('Few: False, r: %d, %s to %s: val_acc: %f, test_acc: %f\n' %
                    (args.r, args.pretrain_dataset, args.test_dataset, max_acc, max_test_acc))
    return {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(max_acc),
        "test_accuracy": float(max_test_acc),
        "best_epoch": int(max_epoch),
        "final_loss": float(loss.item()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc) if auc is not None else None,
        "ap": float(ap) if ap is not None else None,
    }



from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from typing import Tuple
from torch import Tensor, device


class GNN(torch.nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn_type='TransformerConv', gnn_layer_num=2):
        super().__init__()
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if gnn_layer_num < 1:
            raise ValueError('GNN layer_num should >=1 but you set {}'.format(gnn_layer_num))
        elif gnn_layer_num == 1:
            self.conv = nn.ModuleList([GraphConv(input_dim, out_dim)])
        elif gnn_layer_num == 2:
            self.conv = nn.ModuleList([GraphConv(input_dim, 2 * out_dim), GraphConv(2 * out_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, 2 * out_dim)]
            for i in range(gnn_layer_num - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim))
            layers.append(GraphConv(2 * out_dim, out_dim))
            self.conv = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for conv in self.conv[0:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
        node_emb = self.conv[-1](x, edge_index)
        return node_emb


class GATConv_lora(GATConv):
    def __init__(self, in_channels: int | Tuple[int, int], out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0, add_self_loops: bool = True,
                 edge_dim: int | None = None, fill_value: float | Tensor | str = 'mean', bias: bool = True, r: int = 32,
                 **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, add_self_loops, edge_dim,
                         fill_value, bias, **kwargs)
        self.r = r

        if isinstance(in_channels, int):
            self.lin_src_a = Linear(in_channels, self.r, bias=False, weight_initializer='glorot')
            self.lin_src_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
            self.lin_dst = self.lin_src
        else:
            self.lin_src_a = Linear(in_channels[0], self.r, bias=False, weight_initializer='glorot')
            self.lin_src_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
            self.lin_dst_a = Linear(in_channels[1], self.r, bias=False, weight_initializer='glorot')
            self.lin_dst_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_dst = nn.Sequential(self.lin_dst_a, self.lin_dst_b)

        self.reset_parameters_lora()

    def reset_parameters_lora(self):
        torch.nn.init.kaiming_normal_(self.lin_src[0].weight)
        torch.nn.init.zeros_(self.lin_src[1].weight)
        torch.nn.init.kaiming_normal_(self.lin_dst[0].weight)
        torch.nn.init.zeros_(self.lin_dst[1].weight)


class GNNLoRA(torch.nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn, gnn_type='GAT', gnn_layer_num=2, r=32):
        super().__init__()
        self.gnn = gnn
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv_lora
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if gnn_layer_num < 1:
            raise ValueError('GNN layer_num should >=1 but you set {}'.format(gnn_layer_num))
        elif gnn_layer_num == 1:
            self.conv = nn.ModuleList([GraphConv(input_dim, out_dim, r=r)])
        elif gnn_layer_num == 2:
            self.conv = nn.ModuleList([GraphConv(input_dim, 2 * out_dim, r=r), GraphConv(2 * out_dim, out_dim, r=r)])
        else:
            layers = [GraphConv(input_dim, 2 * out_dim, r=r)]
            for i in range(gnn_layer_num - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim, r=r))
            layers.append(GraphConv(2 * out_dim, out_dim, r=r))
            self.conv = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for i in range(self.gnn_layer_num - 1):
            conv1 = self.gnn.conv[i]
            conv2 = self.conv[i]
            x = conv1(x, edge_index) + conv2(x, edge_index)
        node_emb1 = self.gnn.conv[-1](x, edge_index)
        node_emb2 = self.conv[-1](x, edge_index)
        return node_emb1 + node_emb2, node_emb1, node_emb2

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import to_dense_adj
import numpy as np
import yaml
from yaml import SafeLoader


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))


def act(act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        return F.leaky_relu
    elif act_type == 'tanh':
        return torch.tanh
    elif act_type == 'relu':
        return F.relu
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'sigmiod':
        return F.sigmoid


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']
    if (name == 'Computers') | (name == 'Photo'):
        return Amazon(path, name, T.NormalizeFeatures())
    else:
        return Planetoid(path, name, transform=T.NormalizeFeatures())



def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class SMMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(SMMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target, ppr=None):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if ppr is None:
                XX = torch.mean(kernels[:batch_size, :batch_size])
            else:
                XX = torch.mean(kernels[:batch_size, :batch_size] * ppr)
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


def get_ppr_matrix(dataset, alpha: float = 0.05):
    A_tilde = to_dense_adj(dataset.edge_index)[0]
    num_nodes = A_tilde.shape[0]
    D_tilde = torch.diag(1/torch.sqrt(A_tilde.sum(dim=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * torch.linalg.inv(torch.eye(num_nodes).to(A_tilde.device) - (1 - alpha) * H)


def get_ppr_weight(test_dataset):
    ppr_matrix = get_ppr_matrix(test_dataset)
    ppr_matrix[ppr_matrix == 0] = ppr_matrix[ppr_matrix != 0].min()
    ppr_matrix = torch.log(1 + 1 / ppr_matrix)
    ppr_weight = ppr_matrix / ppr_matrix.sum(1).unsqueeze(1) * ppr_matrix.shape[0]
    return ppr_weight


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")



def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def batched_gct_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, mask, tau = 0.5):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / tau)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        idx = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(sim(z1[idx], z1))  # [B, N]
        between_sim = f(sim(z1[idx], z2))  # [B, N]

        losses.append(-torch.log(
            (mask[i * batch_size:(i + 1) * batch_size] * between_sim).sum(1)
            / (refl_sim.sum(1) + between_sim.sum(1)
               - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
    return torch.cat(losses)


def batched_smmd_loss(z1: torch.Tensor, z2, MMD, ppr_weight, batch_size):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        ppr = ppr_weight[mask][:, mask]
        target = next(iter(z2))
        losses.append(MMD(z1[mask], target, ppr))

    return torch.stack(losses).mean()


def get_few_shot_mask(data, shot, dataname, device):
    np.random.seed(0)
    class_num = max(data.y) + 1
    y = data.y.cpu()
    selected = []
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        for i in range(class_num):
            selected.append(np.random.choice(torch.arange(len(y))[(y == i) & train_mask.cpu()], shot))
        train_mask = torch.zeros(len(y)).bool().to(device)
        train_mask[np.concatenate(selected)] = True
    else:
        for i in range(class_num):
            selected.append(np.random.choice(torch.arange(len(y))[y.cpu() == i], shot))
        train_mask = torch.zeros(len(y)).bool().to(device)
        val_mask = torch.zeros(len(y)).bool().to(device)
        test_mask = torch.zeros(len(y)).bool().to(device)
        train_mask[np.concatenate(selected)] = True
        index = np.arange(len(y))[~train_mask.cpu()]
        np.random.shuffle(index)
        val_mask[index[:int(len(index) * 0.2)]] = True
        test_mask[index[int(len(index) * 0.2):]] = True
    return train_mask, val_mask, test_mask


def get_parameter(args):
    config = yaml.load(open(args.para_config), Loader=SafeLoader)
    if args.few:
        if args.shot == 10:
            setting = '10shot'
        else:
            setting = '5shot'
    else:
        setting = 'public'
    args.wd1 = float(config[setting][args.test_dataset]['wd1'])
    args.wd2 = float(config[setting][args.test_dataset]['wd2'])
    args.wd3 = float(config[setting][args.test_dataset]['wd3'])
    args.lr1 = float(config[setting][args.test_dataset]['lr1'])
    args.lr2 = float(config[setting][args.test_dataset]['lr2'])
    args.lr3 = float(config[setting][args.test_dataset]['lr3'])
    args.l1 = float(config[setting][args.test_dataset]['l1'])
    args.l2 = float(config[setting][args.test_dataset]['l2'])
    args.l3 = float(config[setting][args.test_dataset]['l3'])
    args.l4 = float(config[setting][args.test_dataset]['l4'])
    args.num_epochs = config[setting][args.test_dataset]['num_epochs']
    return args


class GraphLoRAWrapped(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes, base_model_path,
                 gnn_type="GCN", num_layers=2, r=8, activation="relu"):
        super().__init__()
        self.base_model_path = base_model_path

        self.gnn_frozen = GNN(in_dim, out_dim, act(activation), gnn_type, num_layers)

        if os.path.exists(base_model_path):
            self.gnn_frozen.load_state_dict(torch.load(base_model_path, map_location='cpu'))
            for p in self.gnn_frozen.parameters():
                p.requires_grad = False
            self.gnn_frozen.eval()
        else:
            # Skip loading — will be trained in pipeline.pretrain()
            print(f"[GraphLoRAWrapped] No checkpoint at {base_model_path}, will train from scratch.")

        self.gnn_lora = GNNLoRA(in_dim, out_dim, act(activation), self.gnn_frozen,
                                gnn_type=gnn_type, gnn_layer_num=num_layers, r=r)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x, edge_index):
        emb, _, _ = self.gnn_lora(x, edge_index)
        logits = self.classifier(emb)
        return F.normalize(logits, p=2, dim=-1)

    def get_embeddings(self, x, edge_index):
        emb, _, _ = self.gnn_lora(x, edge_index)
        return emb

    def reset_with_input_dim(self, new_in_dim: int):
        """
        Rebuild gnn_frozen, gnn_lora, and classifier to accept a new input feature size.
        Keeps the same out_dim, num_classes, gnn_type, layers, r, and activation.
        """
        # Save params you’ll need
        out_dim = self.classifier.out_features
        num_classes = self.classifier.out_features
        gnn_type = self.gnn_frozen.gnn_type
        num_layers = self.gnn_frozen.gnn_layer_num
        activation = self.gnn_frozen.activation
        r = self.gnn_lora.conv[0].lin_src[0].out_channels if hasattr(self.gnn_lora.conv[0], "lin_src") else 8

        # Rebuild modules with new input dimension
        self.gnn_frozen = GNN(new_in_dim, out_dim, activation, gnn_type, num_layers)
        self.gnn_lora = GNNLoRA(new_in_dim, out_dim, activation, self.gnn_frozen,
                                gnn_type=gnn_type, gnn_layer_num=num_layers, r=r)
        self.classifier = nn.Linear(out_dim, num_classes).to(self.classifier.weight.device)

