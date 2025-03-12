import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    """
    A flexible Evaluator class that abstracts away the general evaluation loop.
    You can inject your own 'evaluate_fn' to define how the evaluation is processed.

    Example usage:

        # 1. Define how to evaluate the model.
        def classification_evaluate_fn(model, data, device):
            model.eval()
            data, labels = data
            data = data.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(data.x, data.edge_index)
                loss = loss_fn(outputs, labels)
                predictions = outputs.argmax(dim=1)
                accuracy = (predictions == labels).float().mean().item()

            return {"loss": loss.item(), "accuracy": accuracy}

        # 2. Create the Evaluator and run evaluation.
        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate(
            eval_data=(test_graph, test_labels),
            evaluate_fn=classification_evaluate_fn
        )
        print("Test results:", metrics)
    """
    def __init__(self, model, device=torch.device('cpu')):
        """
        Args:
            model: The PyTorch model to be evaluated.
            device: The device (CPU or GPU).
        """
        self.model = model.to(device)
        self.device = device

    def evaluate(self, eval_data, evaluate_fn):
        """
        Evaluates the model.

        Args:
            eval_data: Dataset for evaluation.
            evaluate_fn: Function to compute evaluation metrics.

        Returns:
            metrics: Computed evaluation metrics.
        """
        if evaluate_fn is None:
            logger.warning("No evaluate_fn provided. Skipping evaluation.")
            return {}

        self.model.eval()
        with torch.no_grad():
            metrics = evaluate_fn(self.model, eval_data, self.device)

        return metrics