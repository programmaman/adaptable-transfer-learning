import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device():
    """
    Returns the best available device: GPU if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("GPU not available, using CPU")
    return device


def compute_clustering_coefficients(data):
    """
    Convert PyG Data object to a NetworkX graph and compute clustering coeffs.
    """
    graph = nx.Graph()
    edges = data.edge_index.t().tolist()  # Convert tensor to list of edges.
    graph.add_edges_from(edges)
    # Ensure isolated nodes are included
    graph.add_nodes_from(range(data.num_nodes))

    clustering = nx.clustering(graph)
    clustering_tensor = torch.tensor([clustering[i] for i in range(data.num_nodes)],
                                     dtype=torch.float)
    return clustering_tensor


def generate_synthetic_graph(num_nodes=200, num_edges=500, feature_dim=16):
    """
    Generate a synthetic graph with random node features and edges.
    Then compute and attach the clustering coefficient as the structural target.
    """
    x = torch.randn((num_nodes, feature_dim))
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    data.structural_targets = compute_clustering_coefficients(data)
    return data


class StructuralFeatureRegressor(nn.Module):
    """
    Simple MLP that maps node embeddings to a single structural value (regression).
    """
    def __init__(self, emb_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings):
        """
        embeddings: [num_nodes, emb_dim]
        return: [num_nodes] (regression output)
        """
        x = F.relu(self.fc1(embeddings))
        x = self.fc2(x)
        return x.squeeze()


def pretrain_node2vec(edge_index,
                      embedding_dim=64,
                      walk_length=20,
                      context_size=10,
                      walks_per_node=10,
                      num_negative_samples=1,
                      batch_size=128,
                      lr=0.01,
                      epochs=10,
                      device=torch.device('cpu')):
    """
    Unsupervised pretraining of Node2Vec embeddings via random walks + negative sampling.
    Returns the trained Node2Vec model.
    """
    # Node2Vec from torch_geometric
    node2vec = Node2Vec(edge_index,
                        embedding_dim=embedding_dim,
                        walk_length=walk_length,
                        context_size=context_size,
                        walks_per_node=walks_per_node,
                        num_negative_samples=num_negative_samples,
                        sparse=True).to(device)

    loader = node2vec.loader(batch_size=batch_size, shuffle=True)
    optimizer = optim.SparseAdam(list(node2vec.parameters()), lr=lr)

    logger.info("Starting Node2Vec unsupervised pretraining...")
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch:03d}, Loss: {total_loss:.4f}")

    return node2vec


def train_structural_feature_regressor(node2vec_model,
                                       data,
                                       epochs=100,
                                       lr=0.01,
                                       weight_decay=0.01,
                                       device=torch.device('cpu')):
    """
    Given Node2Vec embeddings (pretrained or scratch),
    train an MLP to regress the clustering coefficients (structural_targets).
    """
    # Fetch node embeddings from Node2Vec (fixed). If you want them trainable,
    # set requires_grad=True, but typically Node2Vec is used "as-is" after unsupervised training.
    node2vec_model.eval()
    with torch.no_grad():
        embeddings = node2vec_model(torch.arange(data.num_nodes, device=device))

    regressor = StructuralFeatureRegressor(emb_dim=embeddings.size(1)).to(device)

    optimizer = optim.Adam(regressor.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    targets = data.structural_targets.to(device)

    logger.info("Starting structural feature regressor training...")
    for epoch in range(epochs):
        regressor.train()
        optimizer.zero_grad()
        out = regressor(embeddings)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return regressor


class NodeClassifier(nn.Module):
    """
    Classification head over frozen Node2Vec embeddings.
    """
    def __init__(self, emb_dim, out_channels=2, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_channels)

    def forward(self, embeddings):
        x = F.relu(self.fc1(embeddings))
        x = self.fc2(x)
        return x


def fine_tune_node_classification(node2vec_model,
                                  data,
                                  task_labels,
                                  epochs=50,
                                  lr=0.01,
                                  weight_decay=0.01,
                                  device=torch.device('cpu')):
    """
    Fine-tune a node classification head on top of frozen Node2Vec embeddings.
    """
    node2vec_model.eval()  # We'll keep node2vec embeddings frozen
    with torch.no_grad():
        embeddings = node2vec_model(torch.arange(data.num_nodes, device=device))

    classifier = NodeClassifier(emb_dim=embeddings.size(1), out_channels=2).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    labels = task_labels.to(device)

    logger.info("Starting fine-tuning for node classification...")
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        out = classifier(embeddings)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Fine-tune Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # Return classifier for evaluation
    return classifier


def evaluate_node_classifier(classifier, node2vec_model, data, labels, device=torch.device('cpu')):
    """
    Evaluate the classification model on the provided labels.
    """
    classifier.eval()
    with torch.no_grad():
        embeddings = node2vec_model(torch.arange(data.num_nodes, device=device))
        out = classifier(embeddings)
        preds = out.argmax(dim=1)
        accuracy = (preds == labels.to(device)).float().mean().item()
    return accuracy


def fine_tune_link_prediction(node2vec_model,
                              data,
                              epochs=50,
                              lr=0.01,
                              weight_decay=0.01,
                              device=torch.device('cpu')):
    """
    Fine-tune a link prediction head (or simply test dot-product scores) on top of
    Node2Vec embeddings. Here we freeze node2vec and only train a "bias" or do direct
    logistic regression on top of the dot product. For simplicity, we'll do a single bias term.
    """
    node2vec_model.eval()
    # Positive edges
    pos_edge_index = data.edge_index.to(device)
    num_pos = pos_edge_index.shape[1]
    num_nodes = data.num_nodes

    # Negative edges: same number as pos edges
    neg_edge_index = torch.randint(0, num_nodes, (2, num_pos), dtype=torch.long, device=device)

    # A single learnable bias parameter for the dot-product score
    bias = nn.Parameter(torch.zeros(1, device=device))
    optimizer = optim.Adam([bias], lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    logger.info("Starting fine-tuning for link prediction with dot-product + bias...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        with torch.no_grad():
            embeddings = node2vec_model(torch.arange(num_nodes, device=device))

        # Compute dot-product scores for positive/negative edges
        pos_scores = torch.sum(embeddings[pos_edge_index[0]] *
                               embeddings[pos_edge_index[1]], dim=-1) + bias
        neg_scores = torch.sum(embeddings[neg_edge_index[0]] *
                               embeddings[neg_edge_index[1]], dim=-1) + bias

        # Labels
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        preds = torch.cat([pos_scores, neg_scores])

        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    logger.info("Finished link prediction fine-tuning.")
    return bias  # Not very meaningful alone, but indicates the 'head' is learned.


def main():
    device = get_device()

    # Step 1: Generate synthetic data
    data = generate_synthetic_graph(num_nodes=200, num_edges=500, feature_dim=16)

    # Generate random labels for node classification
    task_labels = torch.randint(0, 2, (data.num_nodes,))

    # ========== 1) Pretrain Node2Vec =============
    node2vec_pretrained = pretrain_node2vec(
        data.edge_index,
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        batch_size=128,
        lr=0.01,
        epochs=10,
        device=device
    )

    # (Optional) Use a structural feature regressor on top of the *frozen*
    # node2vec embeddings. This step mimics "pretraining" on structural features
    # if you prefer to see how well the embeddings predict e.g. clustering coefficient.
    structural_regressor = train_structural_feature_regressor(node2vec_pretrained,
                                                              data,
                                                              epochs=100,
                                                              lr=0.01,
                                                              device=device)

    # ========== 2) Fine-Tune for Node Classification =============
    pretrained_classifier = fine_tune_node_classification(
        node2vec_model=node2vec_pretrained,
        data=data,
        task_labels=task_labels,
        epochs=50,
        lr=0.01,
        device=device
    )
    acc_pretrained = evaluate_node_classifier(pretrained_classifier,
                                              node2vec_pretrained,
                                              data,
                                              task_labels,
                                              device)
    logger.info(f"Accuracy with Node2Vec (Pretrained) on Node Classification: {acc_pretrained:.4f}")

    # ========== 3) Fine-Tune for Link Prediction =============
    fine_tune_link_prediction(
        node2vec_model=node2vec_pretrained,
        data=data,
        epochs=50,
        lr=0.01,
        device=device
    )

    # ========== (Optional) Compare to "Scratch" Node2Vec =============
    # If you want to see a baseline of "scratch" training, you can re-initialize Node2Vec
    # with the same hyperparams but skip the structural regressor or compare.
    node2vec_scratch = pretrain_node2vec(
        data.edge_index,
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        batch_size=128,
        lr=0.01,
        epochs=10,
        device=device
    )
    scratch_classifier = fine_tune_node_classification(
        node2vec_model=node2vec_scratch,
        data=data,
        task_labels=task_labels,
        epochs=50,
        lr=0.01,
        device=device
    )
    acc_scratch = evaluate_node_classifier(scratch_classifier,
                                           node2vec_scratch,
                                           data,
                                           task_labels,
                                           device)
    logger.info(f"Accuracy with Node2Vec (Scratch) on Node Classification: {acc_scratch:.4f}")

    logger.info("Comparison of pretrained vs scratch Node2Vec completed.")


if __name__ == '__main__':
    main()
