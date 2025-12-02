import copy
import logging
import os
import random
import time
from abc import ABC
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score

from experiments.experiment_utils import EvaluationResult, sample_negative_edges, split_edges_for_link_prediction
from tasks.task import Pretrain

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class TaskPipeline:
    """
    A task-driven pipeline for GNN models with modular data preprocessing.
    """

    def __init__(self, seed=None, device=None, preprocessors: list = None):
        self.seed = self._initialize_seed(seed)
        self.device = self._initialize_device(device)
        self.metadata = {"seed": self.seed}

        self.preprocessors = preprocessors or []

        self._validate_device()
        logger.info(f"Pipeline initialized | seed={self.seed} | device={self.device}")

    # ----------------------------------------------------------------------
    def run(
        self,
        model: torch.nn.Module,
        data,
        tasks: Iterable,
        pretrain_tasks: Iterable = None,
        pretrain_data=None,
        pretrained_snapshot_path="pretrained_snapshot.pt"
    ):
        if model is None:
            raise ValueError("Pipeline.run(): you must provide a model.")
        if not tasks:
            raise ValueError("Pipeline.run(): no tasks provided.")

        self._set_seed()
        start_total = time.time()

        # --------------------------------------------------------------
        # Preprocess target data using DataProcessors
        # --------------------------------------------------------------
        data = self._apply_preprocessors(data).to(self.device)

        # Pretrain dataset selection
        if pretrain_data is None:
            logger.info("Structural Pretraining: Using target dataset for pretraining.")
            pretrain_data = data
        else:
            logger.info("Transfer Learning: Using separate pretrain dataset with same structural pretraining objective.")
            pretrain_data = self._apply_preprocessors(pretrain_data).to(self.device)

        # --------------------------------------------------------------
        # PRETRAINING (with iterable)
        # --------------------------------------------------------------
        if pretrain_tasks:
            logger.info(f"Pretraining tasks...")  # no length
            for pretrain_task in pretrain_tasks:
                logger.info(f"--- Pretraining Task: {pretrain_task.name} ---")
                pretrainer = Pretrain(pretrain_task)
                model, _ = pretrainer.run(model, pretrain_data)

        # Save the pretrained model
        torch.save(model.state_dict(), pretrained_snapshot_path)

        # --------------------------------------------------------------
        # PER-TASK EXECUTION
        # --------------------------------------------------------------
        results = {}

        for task in tasks:
            logger.info(f"\n===== Running Task: {task.name} =====")

            model_copy = copy.deepcopy(model)
            model_copy.load_state_dict(
                torch.load(pretrained_snapshot_path, map_location=self.device)
            )

            task_data = task.prepare(data).to(self.device)
            model_copy = task.train(model_copy, task_data)
            result = task.evaluate(model_copy, task_data)

            result.metadata.update(self.metadata)
            result.metadata.update(task.metadata)

            results[task.name] = result

        self.metadata["total_time"] = time.time() - start_total
        return model, results

    # ----------------------------------------------------------------------
    # INTERNAL: apply all registered DataProcessors
    # ----------------------------------------------------------------------
    def _apply_preprocessors(self, data):
        for processor in self.preprocessors:
            data = processor(data)
        return data

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
        raise NotImplementedError("Subclass must implement train_link_prediction")

    def evaluate_link_prediction(self, model, data, rem_edge_list):
        raise NotImplementedError("Subclass must implement evaluate_link_prediction")




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


