import copy
import logging
import random
import time
from typing import Iterable

import numpy as np
import torch
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

