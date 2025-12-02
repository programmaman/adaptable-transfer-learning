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


