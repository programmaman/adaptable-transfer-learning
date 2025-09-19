import logging
import time
import torch
import random
import numpy as np
from abc import ABC, abstractmethod
from experiments.experiment_utils import split_edges_for_link_prediction, EvaluationResult

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class Pipeline(ABC):
    def __init__(self, seed=None, device=None, train_ratio=0.6, val_ratio=0.2):
        # Always have a seed: if user didn't pass one, derive from current time
        self.seed = seed if seed is not None else int(time.time() * 1e6) % (2**32)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.metadata = {"seed": self.seed}
        logger.info(f"Pipeline initialized with seed {self.seed} on device {self.device}")
        if self.device != "cuda":
            logger.warning("CUDA not available, using CPU. This may be slow!")


    def run(self, data, labels, pretrain_epochs=100, finetune_epochs=30):
        logger.info("Starting pipeline run...")
        logger.info(f"Data has {data.num_nodes} nodes and {data.num_edges} edges.")
        logger.info(f"Labels have {len(labels.unique())} unique classes.")
        logger.info(f"Pretraining for {pretrain_epochs} epochs, Finetuning for {finetune_epochs} epochs.")

        self._set_seed()
        start = time.time()

        data = self.prepare_data(data)
        model = self.initialize_model(data, labels)

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

    # ---- Standardized Seeding ----
    def _set_seed(self):
        # Use the unified seed to control torch, numpy, and Python RNGs
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

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

    # ---- Abstract Methods ----
    @abstractmethod
    def initialize_model(self, data, labels): ...
    @abstractmethod
    def pretrain(self, model, data, epochs): ...
    @abstractmethod
    def finetune_classification(self, model, data, labels, epochs): ...
    @abstractmethod
    def evaluate_classification(self, model, data, labels) -> EvaluationResult: ...
    @abstractmethod
    def finetune_link_prediction(self, model, data, rem_edge_list, epochs): ...
    @abstractmethod
    def evaluate_link_prediction(self, model, data, rem_edge_list) -> EvaluationResult: ...
