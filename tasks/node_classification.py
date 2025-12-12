import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from utilities.experiment_utils import EvaluationResult
from tasks.task import Task


class NodeClassificationTask(Task):
    """
    Node classification task for models following the PyG signature: model(x, edge_index).
    """

    def __init__(self, name="classification", epochs=30, learning_rate=0.01,
                 weight_decay=5e-4, log_every=10):
        super().__init__(name, epochs=epochs)
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
        required_attributes = ["x", "y", "edge_index", "train_mask", "val_mask", "test_mask"]
        for attr in required_attributes:
            assert hasattr(data, attr), f"ClassificationTask.prepare: data is missing required attribute '{attr}'."

        number_of_nodes = data.x.size(0)

        assert data.y.size(0) == number_of_nodes, (
            f"ClassificationTask.prepare: data.y has shape {data.y.shape} but "
            f"data.x has shape {data.x.shape}."
        )
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(data, mask_name)
            assert mask.size(0) == number_of_nodes, (
                f"ClassificationTask.prepare: {mask_name} has shape {mask.shape} but expected shape ({number_of_nodes},)."
            )

        for mask_name in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(data, mask_name)
            assert mask.dtype == torch.bool, f"ClassificationTask.prepare: {mask_name} must be boolean, got {mask.dtype}."


        mask_overlap = (data.train_mask & data.val_mask) | (data.train_mask & data.test_mask) | (
                    data.val_mask & data.test_mask)
        assert not mask_overlap.any(), "ClassificationTask.prepare: train_mask, val_mask, and test_mask must not overlap."
        assert data.train_mask.sum().item() > 0, "ClassificationTask.prepare: train_mask contains zero nodes."

        labels = data.y[data.train_mask]
        assert not torch.is_floating_point(
            data.y), f"ClassificationTask.prepare: data.y must be integer, got {data.y.dtype}."
        assert (labels >= 0).all(), "ClassificationTask.prepare: negative class indices found in data.y."

        assert labels.unique().numel() > 1, "ClassificationTask.prepare: Training set must contain at least two distinct classes."

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
