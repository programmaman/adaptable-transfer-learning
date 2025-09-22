import logging
import random
import time
from abc import ABC

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score

from experiments.experiment_utils import EvaluationResult, split_edges_for_link_prediction

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def _initialize_device(device):
    return device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _initialize_seed(seed):
    if seed is not None:
        return seed
    generated_seed = int(time.time() * 1e6) % (2 ** 32)
    logger.warning("No seed provided. Using time-based seed for reproducibility.")
    return generated_seed


class Pipeline(ABC):
    def __init__(self, seed=None, device=None, train_ratio=0.6, val_ratio=0.2):
        self.seed = _initialize_seed(seed)
        self.device = _initialize_device(device)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.metadata = {"seed": self.seed}

        logger.info(f"Pipeline initialized with seed {self.seed} on device {self.device}")
        #Check torch device is cuda if available
        if torch.cuda.is_available() and self.device.type != 'cuda':
            logger.warning("CUDA is available but the device is not set to 'cuda'. This may lead to suboptimal performance.")

    def run(self, data, labels, model, pretrain_epochs=100, finetune_epochs=30):
        """
        Runs the pipeline on the provided model.
        """
        if model is None:
            raise ValueError("You must pass a model instance to run().")

        logger.info("Starting pipeline run...")
        logger.info(f"Data has {data.num_nodes} nodes and {data.num_edges} edges.")
        logger.info(f"Labels have {len(labels.unique())} unique classes.")
        logger.info(f"Pretraining for {pretrain_epochs} epochs, Finetuning for {finetune_epochs} epochs.")

        self._set_seed()
        start = time.time()

        data = self.prepare_data(data)

        # Phase 1: Pretraining
        model = self.pretrain(model, data, pretrain_epochs)

        # Phase 2: Classification
        classifier_train_start = time.time()
        model = self.finetune_classification(model, data, labels, finetune_epochs)
        self.metadata["classifier_time"] = time.time() - classifier_train_start

        classification_results = self.evaluate_classification(model, data, labels)

        # Phase 3: Link Prediction
        data.edge_index, rem_edge_list = split_edges_for_link_prediction(data.edge_index)
        lp_train_start = time.time()
        model = self.finetune_link_prediction(model, data, rem_edge_list, finetune_epochs)
        self.metadata["link_pred_time"] = time.time() - lp_train_start

        link_pred_results = self.evaluate_link_prediction(model, data, rem_edge_list)

        self.metadata["total_time"] = time.time() - start
        classification_results.metadata.update(self.metadata)
        link_pred_results.metadata.update(self.metadata)

        return model, classification_results, link_pred_results

    def transfer_learning_run(self, source_data, source_labels, target_data, target_labels,
                              model, pretrain_epochs=100, finetune_epochs=30):
        """
        Runs transfer learning:
          - Pretrain on source dataset
          - Finetune + evaluate on target dataset
        """
        if model is None:
            raise ValueError("You must pass a model instance to transfer_run().")

        logger.info("Starting transfer pipeline run...")
        self._set_seed()
        start = time.time()

        # ---- Source Pretraining ----
        logger.info(f"Source graph: {source_data.num_nodes} nodes, {source_data.num_edges} edges")
        source_data = self.prepare_data(source_data)
        model = self.pretrain(model, source_data, pretrain_epochs)

        # ---- Target Fine-tuning ----
        logger.info(f"Target graph: {target_data.num_nodes} nodes, {target_data.num_edges} edges")
        target_data = self.prepare_data(target_data)

        classifier_train_start = time.time()
        model = self.finetune_classification(model, target_data, target_labels, finetune_epochs)
        self.metadata["classifier_time"] = time.time() - classifier_train_start

        classification_results = self.evaluate_classification(model, target_data, target_labels)

        # ---- Target Link Prediction ----
        target_data.edge_index, rem_edge_list = split_edges_for_link_prediction(target_data.edge_index)
        lp_train_start = time.time()
        model = self.finetune_link_prediction(model, target_data, rem_edge_list, finetune_epochs)
        self.metadata["link_pred_time"] = time.time() - lp_train_start

        link_pred_results = self.evaluate_link_prediction(model, target_data, rem_edge_list)

        # ---- Metadata + Return ----
        self.metadata["total_time"] = time.time() - start
        classification_results.metadata.update(self.metadata)
        link_pred_results.metadata.update(self.metadata)

        return model, classification_results, link_pred_results

    # ---- Default Train/Val/Test Split ----
    def prepare_data(self, data):
        """Creates train/val/test masks using the pipeline's seed for reproducibility."""
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)  # uses the seeded RNG
        train_cut = int(self.train_ratio * num_nodes)
        val_cut = int((self.train_ratio + self.val_ratio) * num_nodes)

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:train_cut]] = True
        data.val_mask[perm[train_cut:val_cut]] = True
        data.test_mask[perm[val_cut:]] = True
        return data

    # ---- Standardized Seeding ----
    def _set_seed(self):
        # Use the unified seed to control torch, numpy, and Python RNGs
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    # ---- Default Methods (Overridable) ----

    def pretrain(self, model, data, epochs):
        """
        Default: skip pretraining and return model unchanged.
        Override if your model has a pretraining phase.
        """
        logger.info("No pretraining implemented; skipping.")
        return model

    def finetune_classification(self, model, data, labels, epochs):
        """
        Default: raise NotImplementedError.
        Subclasses should override with training loop.
        """
        raise NotImplementedError("finetune_classification() must be implemented by a subclass.")

    def evaluate_classification(self, model, data, labels) -> EvaluationResult:
        """
        Default: raise NotImplementedError.
        Subclasses should override with evaluation logic.
        """
        raise NotImplementedError("evaluate_classification() must be implemented by a subclass.")

    def finetune_link_prediction(self, model, data, rem_edge_list, epochs):
        """
        Default: no link prediction fine-tuning (identity).
        Override if you have a LP-specific training loop.
        """
        logger.info("No link prediction fine-tuning implemented; skipping.")
        return model

    def evaluate_link_prediction(self, model, data, rem_edge_list) -> EvaluationResult:
        """
        Default: raise NotImplementedError.
        Subclasses should override with LP evaluation.
        """
        raise NotImplementedError("evaluate_link_prediction() must be implemented by a subclass.")



from experiments.experiment_utils import EvaluationResult, sample_negative_edges


class DefaultPipeline(Pipeline):
    """
    A default pipeline implementation of the abstract Pipeline.
    Provides standard classification and link prediction logic
    that can be used with GNN-style models (e.g., SimpleGNN, SimpleGAT).
    """

    # ------------------------------
    # Fine-tune for Node Classification
    # ------------------------------
    def finetune_classification(self, model, data, labels, epochs=30,
                                lr=0.01, weight_decay=5e-4, log_every=10):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        logger.info("Fine-tuning on Node Classification")
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data.x.to(self.device), data.edge_index.to(self.device))
            loss = criterion(out[data.train_mask], labels[data.train_mask].to(self.device))
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                val_result = self.evaluate_classification(model, data, labels, mask=data.val_mask)
                logger.info(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val Acc {val_result.accuracy:.4f}")
        return model

    # ------------------------------
    # Evaluate Classification
    # ------------------------------
    def evaluate_classification(self, model, data, labels, mask=None, verbose=False) -> EvaluationResult:
        if mask is None:
            mask = data.test_mask

        model.eval()
        with torch.no_grad():
            out = model(data.x.to(self.device), data.edge_index.to(self.device))
            preds = out.argmax(dim=1)[mask].cpu()
            trues = labels[mask].cpu()

        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds, average='macro', zero_division=0)
        recall = recall_score(trues, preds, average='macro', zero_division=0)
        f1 = f1_score(trues, preds, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(trues, preds, multi_class='ovr', average='macro')
        except ValueError:
            auc = None

        if verbose:
            logger.info(f"Acc {acc:.4f} | Precision {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f} | AUC {auc}")

        return EvaluationResult(
            accuracy=acc, precision=precision, recall=recall,
            f1=f1, auc=auc, preds=preds
        )

    # ------------------------------
    # Fine-tune for Link Prediction
    # ------------------------------
    def finetune_link_prediction(self, model, data, rem_edge_list, epochs=30,
                                 lr=0.01, weight_decay=5e-4, log_every=10):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        pos_edges = rem_edge_list[0][0]
        n = data.num_nodes

        def score(u, v):
            return (u * v).sum(dim=1)

        logger.info("Fine-tuning on Link Prediction")
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            emb = model(data.x.to(self.device), data.edge_index.to(self.device))
            neg_edges = sample_negative_edges(pos_edges, n).to(self.device)

            pos_scores = score(emb[pos_edges[:, 0]], emb[pos_edges[:, 1]])
            neg_scores = score(emb[neg_edges[:, 0]], emb[neg_edges[:, 1]])

            logits = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

            loss = bce_loss(logits, labels)
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                logger.info(f"Epoch {epoch:03d} | LP Loss {loss.item():.4f}")
        return model

    # ------------------------------
    # Evaluate Link Prediction
    # ------------------------------
    def evaluate_link_prediction(self, model, data, rem_edge_list, verbose=False) -> EvaluationResult:
        model.eval()
        with torch.no_grad():
            emb = model(data.x.to(self.device), data.edge_index.to(self.device))

        pos_edges = rem_edge_list[0][0]
        n = data.num_nodes
        neg_edges = sample_negative_edges(pos_edges, n).to(self.device)

        def score(u, v):
            return (u * v).sum(dim=1)

        pos_scores = score(emb[pos_edges[:, 0]], emb[pos_edges[:, 1]])
        neg_scores = score(emb[neg_edges[:, 0]], emb[neg_edges[:, 1]])

        scores = torch.cat([pos_scores, neg_scores]).cpu()
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu()
        preds = (scores > 0).float()

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        if verbose:
            logger.info(
                f"LP Eval | Acc {acc:.4f} | Prec {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | AP {ap:.4f}")

        return EvaluationResult(
            accuracy=acc, precision=precision, recall=recall,
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
        return model

    # ------------------------------
    # Fine-tune for Node Classification
    # ------------------------------
    def finetune_classification(
            self, model, data, labels, epochs=30, lr=0.01, weight_decay=5e-4, log_every=10
    ):
        assert model.num_classes is not None, "Model must define num_classes for classification."
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        logger.info("Fine-tuning StructuralGNN on Node Classification")
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            embeddings = model(
                data.x.to(self.device),
                data.edge_index.to(self.device)
            )
            logits = model.classify_nodes(embeddings)

            # --- DEBUG LOGGING ---
            y_train = labels[data.train_mask]
            logger.info(f"[Epoch {epoch:03d}] logits shape: {logits.shape}, "
                        f"num_classes={model.num_classes}")
            logger.info(f"[Epoch {epoch:03d}] labels shape: {y_train.shape}, "
                        f"unique labels={torch.unique(y_train)} "
                        f"(min={y_train.min().item()}, max={y_train.max().item()})")
            logger.info(f"[Epoch {epoch:03d}] train_mask sum: {data.train_mask.sum().item()}")

            loss = criterion(logits[data.train_mask], y_train.to(self.device))
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

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        if verbose:
            logger.info(f"LP Eval | Acc {acc:.4f} | Prec {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | AP {ap:.4f}")

        return EvaluationResult(acc, precision, recall, f1, auc, ap, preds)


