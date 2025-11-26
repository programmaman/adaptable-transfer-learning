import copy
import logging
import os
import random
import time
from abc import ABC
from typing import Iterable

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from torch import nn, functional

from experiments.experiment_utils import EvaluationResult, sample_negative_edges, split_edges_for_link_prediction

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def _initialize_device(device):
    if device is None:
        chosen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        chosen = device if isinstance(device, torch.device) else torch.device(device)
    logger.info(f"Selected device: {chosen}")
    if torch.cuda.is_available() and getattr(chosen, "type", None) != "cuda":
        logger.warning("CUDA is available but the selected device is not 'cuda'. This may lead to suboptimal performance.")
    return chosen


def _initialize_seed(seed):
    if seed is not None:
        return seed
    generated_seed = int(time.time() * 1e6) % (2 ** 32)
    logger.warning("No seed provided. Using time-based seed for reproducibility.")
    return generated_seed


class TaskPipeline:
    """
    A task-driven pipeline for GNN models.

    Core lifecycle:
        - prepare_data()
        - pretrain()
        - for each Task:
            - task.prepare()
            - task.train()
            - task.evaluate()
    """

    def __init__(self, seed=None, device=None):
        self.seed = self._initialize_seed(seed)
        self.device = self._initialize_device(device)
        self.metadata = {"seed": self.seed}

        self._validate_device()
        logger.info(f"Pipeline initialized | seed={self.seed} | device={self.device}")

    # ----------------------------------------------------------------------
    # MAIN ENTRYPOINT
    # ----------------------------------------------------------------------
    def run(
        self,
        model: torch.nn.Module,
        data,
        tasks: Iterable,
        pretrain_data=None,
        pretrain_epochs=0,
        pretrained_snapshot_path="pretrained_snapshot.pt"
    ):
        """
        Orchestrates the full lifecycle of tasks in a modular way.
        """

        if model is None:
            raise ValueError("Pipeline.run(): you must provide a model.")
        if not tasks:
            raise ValueError("Pipeline.run(): no tasks provided.")

        self._set_seed()

        start_total = time.time()

        # ------------------------------------------------------------------
        # Prepare Target Data
        # ------------------------------------------------------------------
        data = self.prepare_data(data).to(self.device)

        # Determine pretraining dataset
        if pretrain_data is None:
            pretrain_data = data
        else:
            logger.info("Transfer Learning: Using separate pretrain dataset.")
            pretrain_data = self.prepare_data(pretrain_data).to(self.device)

        # ------------------------------------------------------------------
        # PRETRAINING
        # ------------------------------------------------------------------
        if pretrain_epochs > 0:
            logger.info(f"Pretraining for {pretrain_epochs} epochs...")
            model = self.pretrain(model, pretrain_data, pretrain_epochs)

        logger.info(f"Saving pretrained snapshot → {pretrained_snapshot_path}")
        torch.save(model.state_dict(), pretrained_snapshot_path)

        # ------------------------------------------------------------------
        # PER-TASK EXECUTION
        # ------------------------------------------------------------------
        results = {}

        for task in tasks:
            logger.info(f"\n===== Running Task: {task.name} =====")

            # Clone model so each task starts from identical state
            model_copy = copy.deepcopy(model)
            model_copy.load_state_dict(
                torch.load(pretrained_snapshot_path, map_location=self.device)
            )

            # Prepare task-specific version of the data
            task_data = task.prepare(data)

            # Move to device
            task_data = task_data.to(self.device)

            # Train task head
            model_copy = task.train(model_copy, task_data)

            # Evaluate
            result = task.evaluate(model_copy, task_data)

            # Merge inherited metadata
            result.metadata.update(self.metadata)
            result.metadata.update(task.metadata)

            results[task.name] = result

        self.metadata["total_time"] = time.time() - start_total

        return model, results

    # ----------------------------------------------------------------------
    # ABSTRACT HOOKS (Overridable)
    # ----------------------------------------------------------------------

    def prepare_data(self, data):
        """Override for data preparation such as masks."""
        return data

    def pretrain(self, model, data, epochs):
        """Default: no pretraining."""
        logger.info("No pretraining implemented, skipping.")
        return model

    # ----------------------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------------------

    @staticmethod
    def _initialize_seed(seed):
        if seed is not None:
            return seed
        generated = int(time.time() * 1e6) % (2**32)
        logger.warning(f"No seed provided — using generated seed {generated}")
        return generated

    @staticmethod
    def _initialize_device(device):
        if device is None:
            chosen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            chosen = device if isinstance(device, torch.device) else torch.device(device)

        logger.info(f"Selected device: {chosen}")
        return chosen

    def _validate_device(self):
        if torch.cuda.is_available() and self.device.type != "cuda":
            logger.warning("CUDA available but pipeline device is CPU — expect slower training.")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)


class Pipeline(ABC):
    """
    Original pipeline implementation:
    - Pretraining (optional)
    - Optional node classification head
    - Optional link prediction head
    """

    def __init__(self, seed=None, device=None, train_ratio=0.6, val_ratio=0.2):
        self.seed = _initialize_seed(seed)
        self.device = _initialize_device(device)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.metadata = {"seed": self.seed}

        self._validate_device()
        logger.info(f"Pipeline initialized with seed {self.seed} on device {self.device}")

    # ----------------------------------------------------------------------
    # MAIN EXECUTION FLOW
    # ----------------------------------------------------------------------
    def run(
        self,
        model,
        data,
        labels=None,
        pretrain_data=None,
        pretrain_epochs=100,
        class_epochs=30,
        lp_epochs=30,
        tasks=None,
        pretrained_path="pretrained_snapshot.pt",
    ):
        """
        Unified execution entry point.
        Performs:
            1. Data prep
            2. Pretraining
            3. Classification (optional)
            4. Link prediction (optional)
        """

        if tasks is None:
            tasks = ["classification", "link_prediction"]

        if model is None:
            raise ValueError("You must pass a model instance to run().")

        self._set_seed()
        start_total = time.time()

        # ------------------------------------------------------------------
        # 1. Prepare target dataset
        # ------------------------------------------------------------------
        target_data = self.prepare_data(data).to(self.device)

        # Determine pretraining dataset
        pretrain_data = pretrain_data if pretrain_data is not None else target_data

        if pretrain_data is not None:
            pretrain_data = self.prepare_data(pretrain_data).to(self.device)
            logger.info("Transfer Learning mode: Pretraining on separate source data.")

        # ------------------------------------------------------------------
        # 2. Pretraining
        # ------------------------------------------------------------------
        logger.info(f"Starting Pretraining ({pretrain_epochs} epochs)...")
        model = self.pretrain(model, pretrain_data, pretrain_epochs)

        logger.info(f"Saving pretrained model snapshot to: {pretrained_path}")
        torch.save(model.state_dict(), pretrained_path)

        results = {}

        # ------------------------------------------------------------------
        # 3. Node Classification
        # ------------------------------------------------------------------
        if "classification" in tasks:
            if labels is None:
                logger.warning("Skipping Classification: No labels provided.")
            else:
                logger.info("Starting Classification Phase...")

                model_copy = copy.deepcopy(model)
                model_copy.load_state_dict(
                    torch.load(pretrained_path, map_location=self.device)
                )

                results["classification"] = self._run_classification(
                    model_copy, target_data, labels, class_epochs
                )

        # ------------------------------------------------------------------
        # 4. Link Prediction
        # ------------------------------------------------------------------
        if "link_prediction" in tasks:
            logger.info("Starting Link Prediction Phase...")

            model_copy = copy.deepcopy(model)
            model_copy.load_state_dict(
                torch.load(pretrained_path, map_location=self.device)
            )

            results["link_prediction"] = self._run_link_prediction(
                model_copy, target_data, lp_epochs
            )

        # ------------------------------------------------------------------
        # Final metadata
        # ------------------------------------------------------------------
        self.metadata["total_time"] = time.time() - start_total

        for key, res in results.items():
            res.metadata.update(self.metadata)

        return model, results

    # ----------------------------------------------------------------------
    # TASK RUNNERS
    # ----------------------------------------------------------------------
    def _run_classification(self, model, data, labels, epochs):
        start = time.time()
        model = self.train_classification(model, data, labels, epochs)
        duration = time.time() - start
        self.metadata["classifier_time"] = duration
        return self.evaluate_classification(model, data, labels)

    def _run_link_prediction(self, model, data, epochs):
        lp_data = data.clone()
        lp_data.edge_index, rem_edge_list = split_edges_for_link_prediction(lp_data.edge_index)

        start = time.time()
        model = self.train_link_prediction(model, lp_data, rem_edge_list, epochs)
        duration = time.time() - start
        self.metadata["link_pred_time"] = duration
        return self.evaluate_link_prediction(model, lp_data, rem_edge_list)

    # ----------------------------------------------------------------------
    # DATA & UTILITY METHODS
    # ----------------------------------------------------------------------
    def prepare_data(self, data):
        """
        Generates masks for classification.
        Assumes data has num_nodes.
        """
        num_nodes = data.num_nodes
        logger.info(f"Preparing data splits for graph with {num_nodes} nodes.")

        perm = torch.randperm(num_nodes)

        train_cut = int(self.train_ratio * num_nodes)
        val_cut = int((self.train_ratio + self.val_ratio) * num_nodes)

        n_train = train_cut
        n_val = val_cut - train_cut
        n_test = num_nodes - val_cut

        logger.info(
            f"Split sizes | Train: {n_train} ({self.train_ratio:.0%}) | "
            f"Val: {n_val} ({self.val_ratio:.0%}) | "
            f"Test: {n_test}"
        )

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:train_cut]] = True
        data.val_mask[perm[train_cut:val_cut]] = True
        data.test_mask[perm[val_cut:]] = True

        return data

    # ----------------------------------------------------------------------
    # DEVICE, SEED, ABSTRACT HOOKS
    # ----------------------------------------------------------------------
    def _validate_device(self):
        if torch.cuda.is_available() and getattr(self.device, "type", "") != "cuda":
            logger.warning("CUDA available but not used. Performance may suffer.")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    # ----------------------------------------------------------------------
    # ABSTRACT METHODS
    # ----------------------------------------------------------------------
    def pretrain(self, model, data, epochs):
        logger.info("Default: Skipping pretraining.")
        return model

    def train_classification(self, model, data, labels, epochs):
        raise NotImplementedError("Subclass must implement train_classification")

    def evaluate_classification(self, model, data, labels):
        raise NotImplementedError("Subclass must implement evaluate_classification")

    def train_link_prediction(self, model, data, rem_edge_list, epochs):
        logger.info("Default: Skipping Link Pred training.")
        return model

    def evaluate_link_prediction(self, model, data, rem_edge_list):
        raise NotImplementedError("Subclass must implement evaluate_link_prediction")



class DefaultPipeline(Pipeline):
    """
    A default pipeline implementation.
    Assumes models follow the standard PyG signature: model(x, edge_index).
    """

    # ------------------------------
    # 1. Classification Training
    # ------------------------------
    # RENAMED from 'finetune_' to 'train_' to match Parent Abstract Class
    def train_classification(self, model, data, labels, epochs=30,
                             lr=0.01, weight_decay=5e-4, log_every=10):

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # OPTIMIZATION: Move to device ONCE, outside the loop
        data = data.to(self.device)
        labels = labels.to(self.device)

        logger.info("Fine-tuning on Node Classification")
        model.train()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

            # Only calculate loss on train_mask
            loss = criterion(out[data.train_mask], labels[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                # We pass the mask explicitly to evaluate on Validation set during training
                val_res = self.evaluate_classification(model, data, labels, mask=data.val_mask)
                logger.info(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val Acc {val_res.accuracy:.4f}")
                model.train()  # Ensure we switch back to train mode after eval

        return model

    # ------------------------------
    # 2. Classification Evaluation
    # ------------------------------
    def evaluate_classification(self, model, data, labels, mask=None, verbose=False):
        # Default to Test Mask if no specific mask is provided
        if mask is None:
            mask = data.test_mask

        data = data.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            # Filter by mask
            logits = logits[mask]
            trues = labels[mask].cpu().numpy()

            # Get Probabilities for AUC
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            # Get Hard Predictions for Acc/F1
            preds = logits.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds, average='macro', zero_division=0)
        recall = recall_score(trues, preds, average='macro', zero_division=0)
        f1 = f1_score(trues, preds, average='macro', zero_division=0)

        # BUG FIX: Use probabilities for AUC, handle multi-class exceptions
        try:
            if len(set(trues)) > 1:  # AUC needs at least 2 classes present
                # Handle binary vs multiclass
                if probs.shape[1] == 2:
                    auc = roc_auc_score(trues, probs[:, 1])  # Binary uses prob of positive class
                else:
                    auc = roc_auc_score(trues, probs, multi_class='ovr', average='macro')
            else:
                auc = 0.0
        except ValueError:
            auc = 0.0

        if verbose:
            logger.info(f"Acc {acc:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")

        return EvaluationResult(
            accuracy=acc, precision=precision, recall=recall,
            f1=f1, auc=auc, preds=preds
        )

    # ------------------------------
    # 3. Link Prediction Training
    # ------------------------------
    def train_link_prediction(self, model, data, rem_edge_list, epochs=30,
                              lr=0.01, weight_decay=5e-4, log_every=10):

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        data = data.to(self.device)
        pos_edges = rem_edge_list[0].to(self.device)
        n = data.num_nodes

        logger.info("Fine-tuning on Link Prediction")
        model.train()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # Get node embeddings
            z = model(data.x, data.edge_index)

            # Negative Sampling (Random edges that don't exist)
            # (Assuming sample_negative_edges is an external helper function)
            neg_edges = sample_negative_edges(pos_edges, num_nodes=n).to(self.device)

            # Decode: Dot product
            pos_scores = (z[pos_edges[0]] * z[pos_edges[1]]).sum(dim=1)
            neg_scores = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=1)

            logits = torch.cat([pos_scores, neg_scores])
            # Labels: 1 for real edges, 0 for fake edges
            edge_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

            loss = bce_loss(logits, edge_labels)
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0:
                logger.info(f"Epoch {epoch:03d} | LP Loss {loss.item():.4f}")

        return model

    # ------------------------------
    # 4. Link Prediction Evaluation
    # ------------------------------
    def evaluate_link_prediction(self, model, data, rem_edge_list, verbose=False):
        model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            z = model(data.x, data.edge_index)

        pos_edges = rem_edge_list[2].to(self.device)
        n = data.num_nodes

        # Sample negatives for evaluation
        neg_edges = sample_negative_edges(pos_edges, num_nodes=n).to(self.device)

        # Decode
        pos_scores = (z[pos_edges[0]] * z[pos_edges[1]]).sum(dim=1)
        neg_scores = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=1)

        scores = torch.cat([pos_scores, neg_scores]).cpu()
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu()

        # Convert logits to probabilities via Sigmoid
        probs = torch.sigmoid(scores)
        preds = (probs > 0.5).float()

        # Metrics
        auc = roc_auc_score(labels, probs)  # Use probabilities for AUC
        ap = average_precision_score(labels, probs)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)

        if verbose:
            logger.info(f"LP Test Results | Acc {acc:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | AP {ap:.4f}")

        return EvaluationResult(
            accuracy=acc, precision=0, recall=0,  # Simplified return
            f1=f1, auc=auc, ap=ap, preds=preds
        )


# pipeline.py (continue after DefaultPipeline)

class TransferLearningPipeline(DefaultPipeline):
    """
    A pipeline that implements transfer learning by pretraining
    the model on a separate dataset (source_data, source_labels)
    before fine-tuning on the target dataset.
    """

    def __init__(self, source_data, source_labels,
                 seed=None, device=None,
                 train_ratio=0.6, val_ratio=0.2):
        super().__init__(seed=seed, device=device,
                         train_ratio=train_ratio, val_ratio=val_ratio)
        self.source_data = source_data.to(self.device)
        self.source_labels = source_labels.to(self.device)

    def pretrain(self, model, data, epochs=50,
                 lr=0.01, weight_decay=5e-4, log_every=10):
        """
        Pretrain the model on a different dataset (self.source_data, self.source_labels).
        This is standard supervised training — NOT structural pretraining.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        logger.info("Pretraining on source dataset (transfer learning).")
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(self.source_data.x, self.source_data.edge_index)
            loss = criterion(out[self.source_data.train_mask],
                             self.source_labels[self.source_data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                val_result = self.evaluate_classification(
                    model, self.source_data, self.source_labels,
                    mask=self.source_data.val_mask
                )
                logger.info(f"[Pretrain Epoch {epoch:03d}] "
                            f"Loss {loss.item():.4f} | Val Acc {val_result.accuracy:.4f}")

        return model


class StructGPipeline(Pipeline):
    """
    A pipeline for StructuralGNN.
    Adds Node2Vec pretraining, classification fine-tuning,
    and link prediction fine-tuning/evaluation.
    """

    # --------------------
    # Phase 1: Pretraining
    # --------------------
    def pretrain(self, model, data, epochs=100, batch_size=128, lr=0.01, verbose=True):
        logger.info("Pretraining Node2Vec embeddings...")
        model.train_node2vec(
            num_epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )
        data = data.to(self.device)
        print("\n=== Phase 2: Pre-training Structural GNN (with internal classifier) ===")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            _, total_loss = model.forward_and_loss(
                data,
                neg_sample_size=5,
                do_node_class=True,
                do_linkpred=True,
                do_featrec=True,
                do_n2v_align=True,
                train_mask=None,
            )
            total_loss.backward()
            optimizer.step()
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"[Pretrain Epoch {epoch:03d}] Total Loss: {total_loss.item():.4f}")

        return model

    # ------------------------------
    # Fine-tune for Node Classification
    # ------------------------------
    def finetune_classification(
            self, model, data, labels, epochs=30, lr=0.01, weight_decay=5e-4, log_every=10
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Attach labels to data so the model can access them internally
        data.y = labels.to(self.device)
        node_indices = torch.arange(data.num_nodes, device=self.device)

        logger.info("Fine-tuning StructuralGNN on Node Classification (internal loss)")
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            # Forward pass through the model
            embeddings = model(
                data.x.to(self.device),
                data.edge_index.to(self.device),
                node_indices
            )

            # Internal loss handles classification logic
            loss = model.node_classification_loss(embeddings, data.y)
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                val_result = self.evaluate_classification(model, data, labels, mask=data.val_mask)
                logger.info(f"[Epoch {epoch:03d}] Loss {loss.item():.4f} | Val Acc {val_result.accuracy:.4f}")

        return model

    def evaluate_classification(self, model, data, labels, mask=None, verbose=True) -> EvaluationResult:
        if mask is None:
            mask = data.test_mask
        model.eval()
        labels = labels.to(self.device)
        with torch.no_grad():
            embeddings = model(data.x.to(self.device), data.edge_index.to(self.device))
            logits = model.classify_nodes(embeddings)
            preds = logits[mask].argmax(dim=1).cpu()
            true = labels[mask].cpu()

        acc = accuracy_score(true, preds)
        precision = precision_score(true, preds, average="macro", zero_division=0)
        recall = recall_score(true, preds, average="macro", zero_division=0)
        f1 = f1_score(true, preds, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(true, preds, multi_class="ovr", average="macro")
        except ValueError:
            auc = None

        if verbose:
            logger.info(f"Eval | Acc {acc:.4f} | Prec {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f} | AUC {auc}")

        return EvaluationResult(acc, precision, recall, f1, auc, preds)

    # ------------------------------
    # Fine-tune for Link Prediction
    # ------------------------------
    def finetune_link_prediction(
            self, model, data, rem_edge_list, epochs=30, lr=0.01, weight_decay=5e-4, neg_sample_size=5, log_every=10
    ):
        logger.info("Fine-tuning StructuralGNN for Link Prediction")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        node_indices = torch.arange(data.num_nodes, device=self.device)

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            embeddings = model(data.x.to(self.device), data.edge_index.to(self.device), node_indices)
            loss = model.link_prediction_loss(
                embeddings, rem_edge_list[0][0].T.to(self.device), neg_sample_size=neg_sample_size
            )
            loss.backward()
            optimizer.step()
            if epoch % log_every == 0 or epoch == epochs:
                logger.info(f"[LP Epoch {epoch:03d}] Loss {loss.item():.4f}")
        return model

    def evaluate_link_prediction(self, model, data, rem_edge_list, verbose=True) -> EvaluationResult:
        model.eval()
        node_indices = torch.arange(data.num_nodes, device=self.device)
        with torch.no_grad():
            gnn_emb = model(data.x.to(self.device), data.edge_index.to(self.device), node_indices)
            n2v_emb = model.node2vec_layer(node_indices)

        pos_edges = rem_edge_list[0][0].to(self.device)
        neg_edges = sample_negative_edges(pos_edges, data.num_nodes).to(self.device)

        def score(u, v):
            return model._pairwise_score(gnn_emb[u], gnn_emb[v], n2v_emb[u], n2v_emb[v]).squeeze()

        pos_scores = score(pos_edges[:, 0], pos_edges[:, 1])
        neg_scores = score(neg_edges[:, 0], neg_edges[:, 1])
        scores = torch.cat([pos_scores, neg_scores]).detach().cpu()
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu()
        preds = (torch.sigmoid(scores) > 0.5).float()


        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        if verbose:
            logger.info(
                f"LP Eval | Acc {acc:.4f} | Prec {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | AP {ap:.4f}")

        return EvaluationResult(acc, precision, recall, f1, auc, ap, preds)


class GraphLoRAPipeline(DefaultPipeline):
    def __init__(self, base_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model_path = base_model_path

    class GraphLoRAPipeline(DefaultPipeline):
        def __init__(self, base_model_path, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.base_model_path = base_model_path

        def pretrain(self, model, data, epochs=100, lr=0.01, weight_decay=5e-4,
                     feat_reduce_dim=256, safety_factor=0.7):
            """
            Memory-aware pretraining with optional feature reduction.

            Args:
                model: GraphLoRAWrapped backbone.
                data: PyG Data object.
                epochs: Number of pretrain epochs.
                lr, weight_decay: Optimizer params.
                feat_reduce_dim: Dim for SVD reduction if needed.
                safety_factor: Fraction of free GPU memory allowed.
            """
            logger.info("Pretraining GraphLoRA backbone with feature reconstruction")

            # --- Step 1: Memory Check and Optional Feature Reduction ---
            number_of_nodes, number_of_features = data.x.size()
            required_bytes = number_of_nodes * number_of_features * 4 * 2  # float32, input + recon
            logger.info(f"Feature matrix size: {number_of_nodes} nodes × {number_of_features} features")
            logger.info(f"Estimated memory required for full reconstruction: {required_bytes / 1e9:.2f} GB")

            use_reduction = False
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0]
                if required_bytes > free_mem * safety_factor:
                    use_reduction = True

            if use_reduction:
                logger.info(f"[Pretrain] Reducing features from {number_of_features} to {feat_reduce_dim}")
                x_cpu = data.x.cpu().numpy()
                svd = TruncatedSVD(n_components=feat_reduce_dim, random_state=self.seed)
                x_reduced = torch.tensor(svd.fit_transform(x_cpu), dtype=torch.float32)
                data = data.__class__(x=x_reduced.to(self.device), edge_index=data.edge_index)
                model.reset_with_input_dim(data.x.size(1))

            decoder = nn.Linear(model.gnn_frozen.conv[-1].out_channels, data.x.size(1)).to(self.device)
            optimizer = torch.optim.Adam(
                list(model.gnn_frozen.parameters()) + list(decoder.parameters()),
                lr=lr, weight_decay=weight_decay
            )

            data = data.to(self.device)

            # --- Step 2: Training Loop (Full-batch) ---
            for epoch in range(epochs):
                model.gnn_frozen.train()
                decoder.train()
                optimizer.zero_grad()

                emb = model.gnn_frozen(data.x, data.edge_index)
                recon = decoder(emb)
                loss = functional.mse_loss(recon, data.x)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"[Pretrain {epoch + 1:03d}] Loss {loss.item():.4f}")

            # --- Step 3: Save Backbone ---
            os.makedirs(os.path.dirname(self.base_model_path), exist_ok=True)
            torch.save(model.gnn_frozen.state_dict(), self.base_model_path)
            logger.info(f"Saved pretrained weights to {self.base_model_path}")
            return model
