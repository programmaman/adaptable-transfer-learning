import copy
import logging
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from tasks.task import Pretrain

logger = logging.getLogger(__name__)


class TaskPipeline:
    """
    A task-driven pipeline for GNN models handling data preprocessing,
    structural pretraining, and downstream task execution.
    """

    def __init__(
            self,
            seed: Optional[int] = None,
            device: Optional[Union[str, torch.device]] = None,
            preprocessors: Optional[List[Any]] = None
    ):
        self.device = self._initialize_device(device)
        self.seed = self._initialize_seed(seed)
        self.preprocessors = preprocessors or []
        self.metadata: Dict[str, Any] = {"seed": self.seed}
        self._validate_device()
        logger.info(f"Pipeline initialized | seed={self.seed} | device={self.device}")

    def run(
            self,
            model: nn.Module,
            data: Any,
            tasks: Iterable[Any],
            pretrain_tasks: Optional[Iterable[Any]] = None,
            pretrain_data: Optional[Any] = None,
            pretrained_snapshot_path: str = "pretrained_snapshot.pt"
    ) -> Tuple[nn.Module, Dict[str, Any]]:

        if model is None:
            raise ValueError("Pipeline.run(): Model cannot be None.")
        if not tasks:
            raise ValueError("Pipeline.run(): No tasks provided.")

        self._set_seed()
        start_time = time.time()

        # Ensure model is on the correct device
        model = model.to(self.device)

        # Preprocess target data
        data = self._apply_preprocessors(data).to(self.device)

        # Determine pretraining dataset
        if pretrain_data is None:
            logger.info("Structural Pretraining: Using target dataset.")
            pretrain_data = data
        else:
            logger.info("Transfer Learning: Using separate pretrain dataset.")
            pretrain_data = self._apply_preprocessors(pretrain_data).to(self.device)

        # Execute Pretraining
        if pretrain_tasks:
            logger.info("Starting pretraining tasks...")
            for task in pretrain_tasks:
                logger.info(f"--- Pretraining Task: {task.name} ---")
                pretrainer = Pretrain(task)
                model, _ = pretrainer.run(model, pretrain_data)

        # Save snapshot (CPU state for portability)
        torch.save(model.cpu().state_dict(), pretrained_snapshot_path)
        model = model.to(self.device)

        # Execute Downstream Tasks
        results = {}
        for task in tasks:
            logger.info(f"Running Task: {task.name}")

            # Initialize fresh model copy from snapshot
            model_copy = copy.deepcopy(model)
            state_dict = torch.load(pretrained_snapshot_path, map_location=self.device)
            model_copy.load_state_dict(state_dict)
            model_copy = model_copy.to(self.device)

            # Prepare task-specific data
            task_data = task.prepare(data).to(self.device)
            logger.info(f"Task {task.name} data prepared.")

            # Train and Evaluate
            model_copy = task.train(model_copy, task_data)
            logger.info(f"Task {task.name} training complete.")
            result = task.evaluate(model_copy, task_data)
            logger.info(f"Task {task.name} evaluation complete.")

            # Update metadata
            result.metadata.update(self.metadata)
            result.metadata.update(task.metadata)
            results[task.name] = result

        # This line is now valid because self.metadata is Dict[str, Any]
        self.metadata["total_time"] = time.time() - start_time
        return model, results

    def _apply_preprocessors(self, data: Any) -> Any:
        for processor in self.preprocessors:
            data = processor(data)
        return data

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _initialize_seed(seed: Optional[int]) -> int:
        if seed is not None:
            return seed
        generated = int(time.time() * 1e6) % (2 ** 32)
        logger.warning(f"No seed provided — using generated seed {generated}")
        return generated

    @staticmethod
    def _initialize_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device if isinstance(device, torch.device) else torch.device(device)

    def _validate_device(self) -> None:
        if torch.cuda.is_available() and self.device.type != "cuda":
            logger.warning("CUDA available but pipeline device is CPU.")