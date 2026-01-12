import unittest
import torch
from pipeline.pipeline import TaskPipeline
from models.model_factory import ModelFactory
from tasks.link_prediction import LinkPredictionTask
from tasks.node_classification import NodeClassificationTask
from utilities.dataloader import load_dataset


class TestModelExecution(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.datasets = ["Cora", "Synthetic"]

    def test_all_models_run_successfully(self):
        """Test that all models can run for 2 epochs without errors"""
        pipeline = TaskPipeline(device=self.device)

        # Create tasks with 2 epochs
        tasks = [
            NodeClassificationTask(epochs=2),
            LinkPredictionTask(epochs=2)
        ]

        all_models = ModelFactory.initialize_all()

        for dataset in self.datasets:
            with self.subTest(dataset=dataset):
                data, labels, info = load_dataset(dataset)

                for model_name, init_func in all_models:
                    with self.subTest(model=model_name):
                        try:
                            model = init_func(data, labels)
                            model_results = pipeline.run(model=model, data=data, tasks=tasks)
                            self.assertIsNotNone(model_results)
                            print(f"✓ {model_name} on {dataset} passed")
                        except NotImplementedError as e:
                            self.skipTest(f"{model_name} not implemented: {e}")


if __name__ == "__main__":
    unittest.main()
