from pipeline.pipeline import TaskPipeline
from models.model_factory import ModelFactory
from tasks.link_prediction import LinkPredictionTask
from tasks.node_classification import NodeClassificationTask
from utilities.dataloader import get_all_dataset_names, load_dataset


def run():
    datasets_config = {
        "Cora": {},
        "CiteSeer": {},
        "PubMed": {},
        "Computers": {},
        "Photo": {},
        "Deezer-Europe": {
            "edge_path": "../datasets/deezer_europe/deezer_europe_edges.csv",
            "features_path": "../datasets/deezer_europe/deezer_europe_features.json",
            "target_path": "../datasets/deezer_europe/deezer_europe_target.csv"
        },
        "Twitch-Gamers": {
            "edge_path": "../datasets/twitch_gamers/large_twitch_edges.csv",
            "meta_path": "../datasets/twitch_gamers/large_twitch_features.csv"
        },
        "MUSAE-Facebook": {
            "edge": "../datasets/facebook_large/musae_facebook_edges.csv",
            "features": "../datasets/facebook_large/musae_facebook_features.json",
            "target": "../datasets/facebook_large/musae_facebook_target.csv"
        },
        "MUSAE-GitHub": {
            "edge": "../datasets/git_web_ml/musae_git_edges.csv",
            "features": "../datasets/git_web_ml/musae_git_features.json",
            "target": "../datasets/git_web_ml/musae_git_target.csv"
        },
        "Email-EU-Core": {
            "edge_path": "../datasets/email-eu-core/email-Eu-core.txt",
            "label_path": "../datasets/email-eu-core/email-Eu-core-department-labels.txt"
        },
        "Synthetic": {}
    }

    pipeline = TaskPipeline()
    tasks = [NodeClassificationTask(), LinkPredictionTask()]

    for dataset, kwargs in datasets_config.items():
        data, labels, info = load_dataset(dataset, **kwargs)
        print(f"Dataset: {dataset}, Info: {info}")
        gat = ModelFactory().initialize_gat(data, labels)
        model, results = pipeline.run(model=gat, data=data, tasks=tasks)
        print(f"Results for {dataset}: {results}")


if __name__ == "__main__":
    run()