from pipeline.pipeline import TaskPipeline
from models.model_factory import ModelFactory
from tasks.link_prediction import LinkPredictionTask
from tasks.node_classification import NodeClassificationTask
from utilities.dataloader import get_all_dataset_names, load_dataset


def run():
    datasets = get_all_dataset_names()
    pipeline = TaskPipeline()
    tasks = [NodeClassificationTask(),LinkPredictionTask()]
    for dataset in datasets:
        data, labels, info = load_dataset(dataset)
        print(f"Dataset: {dataset}, Info: {info}")
        gat = ModelFactory().initialize_gat(data, labels)
        print(f"Initialized GAT model for {dataset}: {gat}")
        model, results = pipeline.run(model=gat, data=data, tasks=tasks)
        print(f"Results for {dataset}: {results}")



if __name__ == "__main__":
    run()