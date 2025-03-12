import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
       A flexible Trainer class that abstracts away the general training loop.
       You can inject your own 'train_step_fn' to define how each batch/epoch is processed.
       Optionally, you can also inject an 'evaluate_fn' for validation or testing.

       Example usage:

           # 1. Define how to compute loss in a single training step.
           def classification_train_step_fn(model, data, device):
               model.train()
               data, labels = data
               data = data.to(device)
               labels = labels.to(device)

               outputs = model(data.x, data.edge_index)
               loss = loss_fn(outputs, labels)
               return loss

           # 2. Define how to evaluate the model (optional).
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

           # 3. Create the Trainer and run training/evaluation.
           trainer = Trainer(model, optimizer, device)
           trainer.train(
               train_data=(train_graph, train_labels),
               train_step_fn=classification_train_step_fn,
               epochs=50
           )
           metrics = trainer.evaluate(
               eval_data=(test_graph, test_labels),
               evaluate_fn=classification_evaluate_fn
           )
           print("Test results:", metrics)
       """
    def __init__(self, model, optimizer, device=torch.device('cpu'), redundant_train_step=10, checkpoint_path="best_model.pth"):
        """
        Args:
            model: The PyTorch model to be trained/evaluated.
            optimizer: The optimizer to be used.
            device: The device (CPU or GPU).
            redundant_train_step: Number of epochs to wait for improvement in validation loss before stopping early.
            checkpoint_path: File path to save the best model checkpoint.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.redundant_train_step = redundant_train_step
        self.checkpoint_path = checkpoint_path

    def train(self, train_data, val_data, train_step_fn, eval_fn, epochs=50, log_interval=10):
        """
        Trains the model with early stopping and checkpointing.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            train_step_fn: Function to perform a training step.
            eval_fn: Function to compute validation loss.
            epochs: Maximum number of training epochs.
            log_interval: Interval for logging training loss.
        """
        logger.info("Starting training...")

        best_val_loss = float("inf")
        redundant_train_step_counter = 0

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Perform training step
            loss = train_step_fn(self.model, train_data, self.device)
            loss.backward()
            self.optimizer.step()

            # Validation Step
            val_loss = eval_fn(self.model, val_data, self.device)

            if epoch % log_interval == 0:
                logger.info(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                redundant_train_step_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)  # Save best model checkpoint
                logger.info(f"Checkpoint saved at epoch {epoch}")
            else:
                redundant_train_step_counter += 1

            if redundant_train_step_counter >= self.redundant_train_step:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Validation Loss: {best_val_loss:.4f}")
                break  # Stop training if no improvement for `redundant_train_step` epochs

        return self.model

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
