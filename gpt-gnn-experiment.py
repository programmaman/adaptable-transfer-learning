import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from gpt_gnn import GPT_GNN, GNN, Classifier

# 1. Generate a random graph
num_nodes = 100
num_edges = 300
num_node_types = 3
num_relations = 2

G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
data = from_networkx(G)

# 2. Create node features
node_feature = torch.randn(num_nodes, 64)  # Assuming 64 input features per node
node_type = torch.randint(0, num_node_types, (num_nodes,))  # Random node types
edge_index = torch.stack([data.edge_index[0], data.edge_index[1]])  # Convert to tensor
edge_type = torch.randint(0, num_relations, (data.edge_index.shape[1],))  # Random edge types
edge_time = torch.randint(0, 10, (data.edge_index.shape[1],))  # Random timestamps

# 3. Define the `GNN` Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gnn = GNN(
    in_dim=64, n_hid=128, num_types=num_node_types, num_relations=num_relations,
    n_heads=2, n_layers=2, dropout=0.2, conv_name='hgt'
).to(device)

# 4. Define an Attribute Decoder (for text features)
attr_decoder = Classifier(n_hid=128, n_out=64).to(device)

# 5. Create a fake `rem_edge_list` for link prediction loss
rem_edge_list = {
    0: {0: torch.randint(0, num_nodes, (10, 2))},  # Fake edges
    1: {1: torch.randint(0, num_nodes, (10, 2))}
}

# 6. Initialize GPT_GNN
model = GPT_GNN(gnn, rem_edge_list, attr_decoder, types=num_node_types, neg_samp_num=5, device=device).to(device)

# 7. Move data to the device
node_feature, node_type, edge_index, edge_type, edge_time = [
    x.to(device) for x in [node_feature, node_type, edge_index, edge_type, edge_time]
]

# 8. Run Forward Pass
output = model(node_feature, node_type, edge_time, edge_index, edge_type)
print("Forward Pass Successful. Output Shape:", output.shape)

# 9. Compute Link Loss (if required)
node_dict = {t: (0, num_nodes) for t in rem_edge_list}
loss, _ = model.link_loss(output, rem_edge_list, rem_edge_list, node_dict=node_dict, target_type=0)
print("Link Loss Computed:", loss.item())
