import argparse
import os
from enum import Enum
from herbert import HerbertExperiment
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import neptune.new as neptune

parser = argparse.ArgumentParser()

parser.add_argument("--model", help="select the model")


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
            num_epochs=5,
            test_size=50,
            val_size=50,
            train_size=100,
            neptune_run=neptune_run,
        ).run()
    else:
        raise NotImplementedError("Model not implemented.")
