# gnn_pipeline/trainer.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .losses import SMMDLoss, batched_contrastive_loss, edge_reconstruction_loss
from .helper import get_ppr_weight

class Trainer:
    """
    Unified GNN trainer for fine-tuning or few-shot adaptation.
    """
    def __init__(self, model, optimizer_cfg, loss_weights, device):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.Adam(
            [
                {"params": model.parameters(), **optimizer_cfg.get("model", {})}
            ]
        )
        self.l1, self.l2, self.l3, self.l4 = [loss_weights[k] for k in ["cls", "smmd", "contrastive", "recon"]]
        self.SMMD = SMMDLoss().to(device)

    def train_epoch(self, data, pretrain_loader=None, tau=0.5):
        self.model.train()
        self.opt.zero_grad()

        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        logits = self.model(x, edge_index)
        labels = data.y.to(self.device)
        mask = data.train_mask

        cls_loss = F.cross_entropy(logits[mask], labels[mask])

        emb = self.model.get_embeddings(x, edge_index)
        smmd_loss = self.SMMD(emb, next(iter(pretrain_loader))) if pretrain_loader else torch.tensor(0.0)
        contrastive = batched_contrastive_loss(emb, emb, labels, tau)
        recon = edge_reconstruction_loss(logits, edge_index, num_nodes=x.size(0))

        total = self.l1 * cls_loss + self.l2 * smmd_loss + self.l3 * contrastive + self.l4 * recon
        total.backward()
        self.opt.step()
        return { "cls": cls_loss.item(), "smmd": smmd_loss.item(), "contrastive": contrastive.item(), "recon": recon.item() }
