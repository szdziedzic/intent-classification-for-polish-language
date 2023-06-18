import argparse
import os
from enum import Enum
from herbert import HerbertExperiment
from torch.optim import Adam
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
            optimizer_class=Adam,
            loss_class=CrossEntropyLoss,
            num_epochs=int(args.num_epochs),
            test_size=int(args.test_size),
            val_size=int(args.val_size),
            train_size=int(args.train_size),
            lr=float(args.lr),
            neptune_run=neptune_run,
            batch_size=int(args.batch_size),
        ).run()
    else:
        raise NotImplementedError("Model not implemented.")
