from abc import ABC
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from experiments.experiment_utils import EvaluationResult
from tasks.task import Task


class NodeClassificationTask(Task, ABC):
    """
    Node classification task for models following the PyG signature: model(x, edge_index).
    """

    def __init__(self, name="classification", epochs=30, learning_rate=0.01,
                 weight_decay=5e-4, log_every=10):
        super().__init__(name)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.log_every = log_every

    # ----------------------------
    # 1. Data Preparation
    # ----------------------------
    def prepare(self, data):
        """
        Validate that the input data is well-formed for node classification.

        Required fields:
            - data.x               (node features)
            - data.y               (node labels)
            - data.edge_index      (graph structure)
            - data.train_mask      (boolean mask)
            - data.val_mask        (boolean mask)
            - data.test_mask       (boolean mask)
        """

        # -------------------------------------------------------------
        # 1. Verify required attributes exist
        # -------------------------------------------------------------
        required_attributes = [
            "x", "y", "edge_index",
            "train_mask", "val_mask", "test_mask"
        ]

        for attribute_name in required_attributes:
            if not hasattr(data, attribute_name):
                raise ValueError(
                    f"ClassificationTask.prepare: Expected data to have attribute "
                    f"'{attribute_name}', but it is missing."
                )

        # -------------------------------------------------------------
        # 2. Verify shapes
        # -------------------------------------------------------------
        number_of_nodes = data.x.size(0)

        # Label count must match number of nodes
        if data.y.size(0) != number_of_nodes:
            raise ValueError(
                f"ClassificationTask.prepare: data.y has shape {data.y.shape} but "
                f"data.x has shape {data.x.shape}. Labels must have one entry per node."
            )

        # Masks must match the number of nodes
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(data, mask_name)
            if mask.size(0) != number_of_nodes:
                raise ValueError(
                    f"ClassificationTask.prepare: {mask_name} has shape {mask.shape} "
                    f"but expected shape ({number_of_nodes},)."
                )

        # -------------------------------------------------------------
        # 3. Masks must be boolean tensors
        # -------------------------------------------------------------
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(data, mask_name)
            if mask.dtype != torch.bool:
                raise TypeError(
                    f"ClassificationTask.prepare: {mask_name} must be a boolean tensor "
                    f"(dtype=torch.bool), but got {mask.dtype}."
                )

        # -------------------------------------------------------------
        # 4. Masks must not overlap
        # -------------------------------------------------------------
        mask_overlap = (
                data.train_mask & data.val_mask |
                data.train_mask & data.test_mask |
                data.val_mask & data.test_mask
        )
        if mask_overlap.any():
            raise ValueError(
                "ClassificationTask.prepare: train_mask, val_mask, and test_mask "
                "must not overlap, but some nodes appear in multiple splits."
            )

        # -------------------------------------------------------------
        # 5. Masks must cover all nodes
        # -------------------------------------------------------------
        all_covered = (
                data.train_mask |
                data.val_mask |
                data.test_mask
        )

        if not all_covered.all():
            missing = (~all_covered).sum().item()
            raise ValueError(
                f"ClassificationTask.prepare: {missing} nodes are not assigned to any mask. "
                "Every node must belong to either train, validation, or test."
            )

        # -------------------------------------------------------------
        # 6. Ensure training set is non-empty
        # -------------------------------------------------------------
        if data.train_mask.sum().item() == 0:
            raise ValueError(
                "ClassificationTask.prepare: train_mask contains zero nodes. "
                "Training requires at least one labeled training node."
            )

        # -------------------------------------------------------------
        # 7. Check that labels on train nodes are valid integers
        # -------------------------------------------------------------
        labels = data.y[data.train_mask]
        if not torch.is_floating_point(data.y):
            # integer labels
            if (labels < 0).any():
                raise ValueError(
                    "ClassificationTask.prepare: Found negative class indices in data.y."
                )
        else:
            raise TypeError(
                "ClassificationTask.prepare: data.y must contain integer class indices, "
                f"but got dtype {data.y.dtype}."
            )

        # -------------------------------------------------------------
        # 8. Optional: ensure there is at least 1 sample per class in training
        # -------------------------------------------------------------
        unique_classes = labels.unique()
        if unique_classes.numel() <= 1:
            raise ValueError(
                "ClassificationTask.prepare: Training set must contain at least two distinct "
                "classes to compute CrossEntropyLoss."
            )

        # -------------------------------------------------------------
        # 9. Done — data is valid
        # -------------------------------------------------------------
        return data

    # ----------------------------
    # 2. Training Loop
    # ----------------------------
    def train(self, model, data):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            logits = model(data.x, data.edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])

            loss.backward()
            optimizer.step()

            if epoch % self.log_every == 0 or epoch == self.epochs:
                val_result = self.evaluate(model, data, mask=data.val_mask)
                self.metadata[f"epoch_{epoch}"] = {
                    "val_accuracy": val_result.accuracy,
                    "loss": loss.item()
                }

        return model

    # ----------------------------
    # 3. Evaluation
    # ----------------------------
    def evaluate(self, model, data, mask=None):
        if mask is None:
            mask = data.test_mask

        model.eval()

        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            logits = logits[mask]
            labels = data.y[mask]

            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)

        # AUC handling (binary or multiclass)
        try:
            if probabilities.shape[1] == 2:
                auc = roc_auc_score(true_labels, probabilities[:, 1])
            else:
                auc = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro')
        except ValueError:
            auc = float('nan')  # AUC cannot be computed

        return EvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            preds=predictions
        )
