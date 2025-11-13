# gnn_pipeline/experiment.py
import torch
from .data import load_dataset, ensure_self_loops, apply_feature_reduction
from .helper import few_shot_masks
from .trainer import Trainer

def run_experiment(model, args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(args.dataset)
    dataset = ensure_self_loops(dataset)
    if args.reduce:
        dataset = apply_feature_reduction(dataset)

    dataset.train_mask, dataset.val_mask, dataset.test_mask = few_shot_masks(dataset, args.shot, args.dataset, device)

    trainer = Trainer(
        model=model,
        optimizer_cfg={
            "model": {"lr": args.lr, "weight_decay": args.wd}
        },
        loss_weights={"cls": args.l1, "smmd": args.l2, "contrastive": args.l3, "recon": args.l4},
        device=device
    )

    for epoch in range(args.epochs):
        losses = trainer.train_epoch(dataset)
        print(f"Epoch {epoch}: {losses}")
