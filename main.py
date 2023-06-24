import argparse
import os
from enum import Enum
from herbert import HerbertExperiment
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import neptune.new as neptune

parser = argparse.ArgumentParser()

parser.add_argument("--model", help="select the model")
parser.add_argument("--num_epochs", help="number of epochs", default=100)
parser.add_argument("--lr", help="learning rate", default=3e-4)
parser.add_argument("--test_size", help="test size", default=None)
parser.add_argument("--val_size", help="validation size", default=None)
parser.add_argument("--train_size", help="train size", default=None)
parser.add_argument("--batch_size", help="batch size", default=32)
parser.add_argument("--num_of_layers", help="number of layers", default=3)
parser.add_argument(
    "--train_base_model", help="train base model", default="false"
)
parser.add_argument("--dropout", help="dropout", default=0.2)


class Models(Enum):
    HERBERT = "herbert"


if __name__ == "__main__":
    neptune_run = None
    if os.environ["NEPTUNE_API_TOKEN"] and os.environ["NEPTUNE_PROJECT"]:
        neptune_run = neptune.init_run(
            project=os.environ["NEPTUNE_PROJECT"],
            api_token=os.environ["NEPTUNE_API_TOKEN"],
        )

    args = parser.parse_args()
    if not args.model:
        raise ValueError("Model not selected. Use --model <model_name> flag.")
    if args.model == Models.HERBERT.value:
        HerbertExperiment(
            optimizer_class=AdamW,
            loss_class=CrossEntropyLoss,
            num_epochs=int(args.num_epochs),
            test_size=int(args.test_size) if args.test_size else None,
            val_size=int(args.val_size) if args.val_size else None,
            train_size=int(args.train_size) if args.train_size else None,
            lr=float(args.lr),
            neptune_run=neptune_run,
            batch_size=int(args.batch_size),
            num_of_layers=int(args.num_of_layers),
            train_base_model=(
                True if args.train_base_model == "true" else False
            ),
            dropout=float(args.dropout),
        ).run()
    else:
        raise NotImplementedError("Model not implemented.")
