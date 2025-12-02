import torch


class DataProcessor:
    def __call__(self, data):
        raise NotImplementedError

class TopologicalSparsify(DataProcessor):
    def __init__(self, keep_ratio=0.8):
        self.keep_ratio = keep_ratio

    def __call__(self, data):
        num_edges = data.edge_index.size(1)
        k = int(num_edges * self.keep_ratio)
        idx = torch.randperm(num_edges)[:k]
        data.edge_index = data.edge_index[:, idx]
        return data


class RobustFeatureNorm(DataProcessor):
    def __call__(self, data):
        x = data.x
        median = x.median(dim=0).values
        mad = (x - median).abs().median(dim=0).values + 1e-6
        data.x = (x - median) / mad
        return data


class AddVirtualNode(DataProcessor):
    def __call__(self, data):
        # Append a virtual node with zero features
        zero = torch.zeros(1, data.x.size(1))
        data.x = torch.cat([data.x, zero], dim=0)

        # Connect virtual node to all nodes
        vn = data.num_nodes
        ones = torch.arange(data.num_nodes)
        edges = torch.stack([ones, torch.full_like(ones, vn)], dim=0)
        rev = edges.flip(0)
        data.edge_index = torch.cat([data.edge_index, edges, rev], dim=1)
        return data


class FeatureQuantization(DataProcessor):
    def __init__(self, bins=16):
        self.bins = bins

    def __call__(self, data):
        mins = data.x.min(dim=0).values
        maxs = data.x.max(dim=0).values
        widths = (maxs - mins) / self.bins
        data.x_bin = ((data.x - mins) / widths).floor().clamp(0, self.bins - 1)
        return data


class AddClassWeights(DataProcessor):
    def __call__(self, data):
        y = data.y
        counts = torch.bincount(y)
        weights = 1.0 / counts[y]
        data.sample_weight = weights
        return data

