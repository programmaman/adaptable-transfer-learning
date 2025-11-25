# python
import copy
import tempfile
from pathlib import Path
import unittest
import torch

from experiments.pipeline import DefaultPipeline, TaskPipeline
from models.gcn import GCN
from tasks.node_classification import NodeClassificationTask


class SimpleData:
    """Minimal stand-in for a PyG Data object used by the pipeline."""
    def __init__(self, x: torch.Tensor, edge_index: torch.LongTensor):
        self.x = x
        self.edge_index = edge_index
        self.num_nodes = x.size(0)

    def clone(self):
        return copy.deepcopy(self)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        for attr in ("train_mask", "val_mask", "test_mask", "y"):
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device))
        return self


class TestDefaultPipelineGCN(unittest.TestCase):
    def test_default_pipeline_classification(self):
        # deterministic behavior
        torch.manual_seed(42)

        # small synthetic graph
        num_nodes = 10
        feat_dim = 4
        x = torch.randn(num_nodes, feat_dim)
        edges = torch.tensor([
            [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9],
            [1,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8]
        ], dtype=torch.long)

        data = SimpleData(x, edges)

        # labels for node classification (2 classes)
        labels = torch.randint(0, 2, (num_nodes,), dtype=torch.long)
        # attach labels to data as some pipelines expect data.y
        data.y = labels.clone()

        # instantiate model and pipeline
        model = GCN(in_channels=feat_dim, out_channels=2)
        pipeline = DefaultPipeline(seed=42, device="cpu", train_ratio=0.6, val_ratio=0.2)

        # Use context manager so tmp dir is cleaned automatically
        with tempfile.TemporaryDirectory() as tmpdir:
            pretrained_path = str(Path(tmpdir) / "pretrained_snapshot.pt")

            # run only classification and keep epochs small for CI speed
            _, results = pipeline.run(
                model=model,
                data=data,
                labels=labels,
                pretrain_epochs=0,
                class_epochs=1,
                lp_epochs=0,
                tasks=["classification"],
                pretrained_path=pretrained_path,
            )

        self.assertIn("classification", results)
        class_res = results["classification"]

        # accept either attribute or dict-style accuracy
        if hasattr(class_res, "accuracy"):
            acc = class_res.accuracy
        elif isinstance(class_res, dict) and "accuracy" in class_res:
            acc = class_res["accuracy"]
        else:
            self.fail("classification result missing accuracy")

        self.assertIsInstance(acc, (float, int, torch.Tensor))
        if isinstance(acc, torch.Tensor):
            acc = acc.item()
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

        print("Evaluation Results:", results)



class TestTaskPipelineGCN(unittest.TestCase):
    def test_task_pipeline_classification(self):
        torch.manual_seed(42)

        # small synthetic graph
        num_nodes = 8
        feat_dim = 3
        x = torch.randn(num_nodes, feat_dim)
        edges = torch.tensor([
            [0, 1, 1, 2, 2, 3, 4, 5],
            [1, 0, 2, 1, 3, 2, 5, 4]
        ], dtype=torch.long)

        data = SimpleData(x, edges)

        # labels for node classification (2 classes) and attach to data
        labels = torch.randint(0, 2, (num_nodes,), dtype=torch.long)
        data.y = labels.clone()

        # create train/val/test masks directly (simple, deterministic)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[:4] = True
        data.val_mask[4:6] = True
        data.test_mask[6:] = True

        # instantiate model and TaskPipeline
        model = GCN(in_channels=feat_dim, out_channels=2)
        pipeline = TaskPipeline(seed=42, device="cpu")

        # the classification task
        cls_task = NodeClassificationTask(
            name="classification",
            epochs=1,        # minimal training for a unit test
            learning_rate=0.01,
            weight_decay=5e-4
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pretrained_path = str(Path(tmpdir) / "task_snapshot.pt")

            # run ONLY classification
            model_out, results = pipeline.run(
                model=model,
                data=data,
                tasks=[cls_task],
                pretrain_epochs=0,
                pretrained_snapshot_path=pretrained_path,
            )

        self.assertIn("classification", results)
        class_res = results["classification"]

        # handle EvaluationResult or dict fallback
        if hasattr(class_res, "accuracy"):
            acc = class_res.accuracy
        elif isinstance(class_res, dict) and "accuracy" in class_res:
            acc = class_res["accuracy"]
        else:
            self.fail("classification result missing accuracy")

        # ensure valid accuracy
        self.assertIsInstance(acc, (float, int, torch.Tensor))
        if isinstance(acc, torch.Tensor):
            acc = acc.item()

        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

        print("Evaluation Results:", results)


if __name__ == "__main__":
    unittest.main()
