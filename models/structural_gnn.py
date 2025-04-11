import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import degree

###############################################################################
# 1. Node2Vec-based Structural Layer
###############################################################################

class StructLayer(nn.Module):
    """
    A thin wrapper around PyG's Node2Vec for modularity.
    """
    def __init__(self, num_nodes, edge_index, embedding_dim=128, walk_length=10,
                 context_size=5, walks_per_node=10, num_negative_samples=1,
                 p=1.0, q=1.0, sparse=True):
        super(StructLayer, self).__init__()
        self.node2vec = pyg_nn.Node2Vec(
            edge_index=edge_index,
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=sparse
        )

    def forward(self, node_indices):
        """
        node_indices: LongTensor of node IDs.
        Returns Node2Vec embeddings for those nodes.
        """
        return self.node2vec(node_indices)

###############################################################################
# 2. Main Model: Improved Structural GNN
###############################################################################

class StructuralGNN(nn.Module):
    """
    A multi-task GNN that:
      1) Uses Node2Vec embeddings (StructLayer) + raw features in a gating mechanism.
      2) Stacks multiple GNN layers (SAGEConv).
      3) Optionally applies a GAT layer for richer attention modeling.
      4) Exposes separate heads for link prediction, node classification,
         and (optionally) feature reconstruction or node2vec alignment.

    You can selectively combine these losses during training for better generalization.
    """
    def __init__(
        self,
        num_nodes,
        edge_index,           # Global edge_index Node2Vec
        input_dim,            # Node feature dimension
        hidden_dim=64,
        output_dim=32,        # Base embedding dimension after the final layer
        embedding_dim=128,    # Node2Vec embedding size
        num_layers=2,
        use_gat=True,
        num_classes=None,     # If None, skip classification head
        feat_reconstruction=False
    ):
        super(StructuralGNN, self).__init__()

        # Node2Vec structural embeddings
        self.node2vec_layer = StructLayer(num_nodes, edge_index, embedding_dim=embedding_dim)

        # A small linear to project Node2Vec embeddings into GNN space
        self.node2vec_proj = nn.Linear(embedding_dim, hidden_dim)

        # Gating over [raw_x, node2vec]
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Stacked GNN (SAGEConv) layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(pyg_nn.SAGEConv(hidden_dim, hidden_dim))

        self.use_gat = use_gat
        if use_gat:
            # Optional GAT layer for final aggregation
            self.gat_layer = pyg_nn.GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        else:
            self.gat_layer = None

        # Final linear to produce "base" embeddings of dimension output_dim
        self.final_sage = pyg_nn.SAGEConv(hidden_dim, output_dim)

        # Classification head (optional)
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(output_dim, num_classes)

        # Feature reconstruction (optional)
        self.feat_reconstruction = feat_reconstruction
        if feat_reconstruction:
            # E.g., a simple MLP that tries to reconstruct the original features
            self.feat_decoder = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

        # Pairwise link predictor (instead of raw cosine) for link prediction
        # We use a small 2-layer MLP that sees [u, v, u*v]
        mlp_in_dim = (output_dim + embedding_dim) * 3
        self.link_predictor = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # scalar score
        )

        # align final embeddings back to Node2Vec
        self.align_proj_gnn = nn.Linear(output_dim, hidden_dim)
        self.align_proj_n2v = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, x, edge_index, node_indices=None):
        """
        Compute final embeddings (size [num_nodes, output_dim]) for the entire graph.
          x: [num_nodes, input_dim]
          edge_index: [2, num_edges]
          node_indices: if None, we produce embeddings for all nodes (0..num_nodes-1).
                        if provided, must be a LongTensor subset.

        For full-batch usage, pass node_indices = range(num_nodes).
        """
        if node_indices is None:
            node_indices = torch.arange(x.size(0), device=x.device)

        # 1) Node2Vec structural embeddings
        with torch.no_grad():
            # Freeze Node2Vec after pretraining, or remove no_grad if you want to fine-tune
            node2vec_emb = self.node2vec_layer(node_indices)

        # 2) Project node2vec and combine with raw x
        node2vec_emb_proj = self.node2vec_proj(node2vec_emb)
        combined = torch.cat([x[node_indices], node2vec_emb_proj], dim=-1)

        gate_weights = self.gate(combined)
        x_proj = self.input_proj(combined)
        # Gating
        raw_x = x[node_indices]
        raw_proj = self.input_proj(torch.cat([
            raw_x,
            torch.zeros(raw_x.size(0), self.node2vec_proj.out_features, device=raw_x.device)
        ], dim=-1))

        gated_x = gate_weights * x_proj + (1 - gate_weights) * raw_proj

        # 3) We need a "full" representation for all nodes. We'll put gated_x
        #    back into the correct shape if node_indices is a subset.
        out = torch.zeros(x.size(0), gated_x.size(-1), device=x.device)
        out[node_indices] = gated_x

        # 4) Pass through GNN layers
        for gnn in self.gnn_layers:
            out = gnn(out, edge_index)
            out = F.relu(out)

        # 5) Optional GAT
        if self.gat_layer is not None:
            out = self.gat_layer(out, edge_index)
            out = F.relu(out)

        # 6) Final linear projection
        # Final projection
        out = self.final_sage(out, edge_index)
        return out

    ###########################################################################
    # 2.1 Node Classification (optional)
    ###########################################################################

    def classify_nodes(self, embeddings):
        """
        Predict class logits for each node.
          embeddings: [num_nodes, output_dim]
        """
        if self.num_classes is None:
            raise ValueError("num_classes is None; classification head not initialized.")
        return self.classifier(embeddings)

    def node_classification_loss(self, embeddings, labels):
        """
        labels: [num_nodes] with integer class IDs.
        """
        logits = self.classify_nodes(embeddings)
        return F.cross_entropy(logits, labels)

    ###########################################################################
    # 2.2 Link Prediction with In-batch Negatives
    ###########################################################################

    def link_prediction_loss(self, embeddings, edge_index, neg_sample_size=25):
        src, dst = edge_index
        num_nodes = embeddings.size(0)

        # Get frozen Node2Vec embeddings
        with torch.no_grad():
            node2vec_emb = self.node2vec_layer(torch.arange(num_nodes, device=embeddings.device))

        # Negative sampling
        neg_dst = torch.randint(0, num_nodes, (neg_sample_size * src.size(0),), device=src.device)
        neg_src = src.repeat(neg_sample_size)

        # Compute scores
        pos_score = self._pairwise_score(
            embeddings[src], embeddings[dst],
            node2vec_emb[src], node2vec_emb[dst]
        )
        neg_score = self._pairwise_score(
            embeddings[neg_src], embeddings[neg_dst],
            node2vec_emb[neg_src], node2vec_emb[neg_dst]
        )

        # Combine for loss
        logits = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ], dim=0)

        return F.binary_cross_entropy_with_logits(logits, labels)

    def _pairwise_score(self, gnn_u, gnn_v, n2v_u, n2v_v):
        """
        Simple MLP over [u, v, u*v] using concatenated GNN and Node2Vec embeddings.
        """
        u = torch.cat([gnn_u, n2v_u], dim=-1)
        v = torch.cat([gnn_v, n2v_v], dim=-1)
        uv_cat = torch.cat([u, v, u * v], dim=-1)
        return self.link_predictor(uv_cat)

    ###########################################################################
    # 2.3 Optional Feature Reconstruction
    ###########################################################################

    def feature_reconstruction_loss(self, embeddings, x):
        """
        If self.feat_reconstruction is True, we decode the learned embedding
        back to the original feature space (x), forcing embeddings to retain
        node-level attribute information.

        x: [num_nodes, input_dim]
        embeddings: [num_nodes, output_dim]
        """
        if not self.feat_reconstruction:
            return torch.tensor(0.0, device=embeddings.device)
        x_recon = self.feat_decoder(embeddings)  # [num_nodes, input_dim]
        return F.mse_loss(x_recon, x)

    ###########################################################################
    # 2.4 Optional Node2Vec Alignment
    ###########################################################################

    def node2vec_alignment_loss(self, embeddings, node2vec_raw):
        """
        Align final GNN embeddings to Node2Vec embeddings with a contrastive
        or cosine-sim-based objective.

        node2vec_raw: [num_nodes, embedding_dim] (Node2Vec space)
        embeddings:   [num_nodes, output_dim]    (GNN space)
        """
        # Project both to same dimension
        n2v_proj = self.align_proj_n2v(node2vec_raw)
        gnn_proj = self.align_proj_gnn(embeddings)

        # Normalize
        n2v_proj = F.normalize(n2v_proj, dim=-1)
        gnn_proj = F.normalize(gnn_proj, dim=-1)

        # Cosine similarity. We want them to be close, so we minimize (1 - cos).
        cos = (n2v_proj * gnn_proj).sum(dim=-1)  # [num_nodes]
        return (1 - cos).mean()

    ###########################################################################
    # 3. Full Forward + Combined Loss Example
    ###########################################################################

    def forward_and_loss(self, data, neg_sample_size=5, do_node_class=False,
                         do_linkpred=False, do_featrec=False,
                         do_n2v_align=False):
        edge_index = data.edge_index
        x = data.x
        embeddings = self.forward(x, edge_index)  # [num_nodes, output_dim]

        total_loss = 0.0

        # (a) Node classification
        if do_node_class and hasattr(data, 'y') and data.y is not None:
            cls_loss = self.node_classification_loss(embeddings, data.y)
            total_loss += cls_loss

        # (b) Link Prediction
        if do_linkpred:
            lp_loss = self.link_prediction_loss(embeddings, edge_index, neg_sample_size)
            total_loss += lp_loss

        # (c) Feature Reconstruction
        if do_featrec:
            fr_loss = self.feature_reconstruction_loss(embeddings, x)
            total_loss += fr_loss

        # (d) Node2Vec alignment
        if do_n2v_align and hasattr(data, 'node2vec_raw') and data.node2vec_raw is not None:
            align_loss = self.node2vec_alignment_loss(embeddings, data.node2vec_raw)
            total_loss += align_loss

        return embeddings, total_loss

    def train_node2vec(self, num_epochs=1, batch_size=128, lr=0.01, verbose=True):
        """
        Pretrain (or continue training) the Node2Vec module.
        Typically, you'd do this before main GNN training.
        """
        node2vec_module = self.node2vec_layer.node2vec
        optimizer = torch.optim.SparseAdam(list(node2vec_module.parameters()), lr=lr)
        loader = node2vec_module.loader(batch_size=batch_size, shuffle=True)

        device = next(node2vec_module.parameters()).device  # <-- Detect model device

        for epoch in range(num_epochs):
            node2vec_module.train()
            total_loss = 0.0
            for pos_rw, neg_rw in loader:
                pos_rw = pos_rw.to(device)  # <-- Move walks to same device
                neg_rw = neg_rw.to(device)

                optimizer.zero_grad()
                loss = node2vec_module.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            if verbose:
                print(f"[Node2Vec Epoch {epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")
