from torch.optim import Optimizer
from torch.nn import Module
from typing import Union
from massive_dataset import MASSIVEDataset
from massive_dataset import MASSIVEDatasetSplitName


class Experiment:
    def __init__(
        self,
        optimizer_class: Optimizer,
        loss_class: Module,
        num_epochs: int = 100,
        num_of_layers: int = 3,
        lr: float = 3e-4,
        test_size: Union[int, None] = None,
        val_size: Union[int, None] = None,
        train_size: Union[int, None] = None,
        batch_size: int = 32,
        neptune_run=None,
        train_base_model: bool = False,
        dropout: float = 0.2,
    ):
        self.name = "Experiment"
        self.optimizer_class = optimizer_class
        self.loss_class = loss_class
        self.test_size = test_size
        self.val_size = val_size
        self.train_size = train_size
        self.batch_size = batch_size
        self.num_of_layers = num_of_layers
        self.train_base_model = train_base_model
        self.dropout = dropout
        self.dataset = MASSIVEDataset()
        self.train_dataloader = self.dataset.get_dataloader(
            MASSIVEDatasetSplitName.TRAIN, self.batch_size, self.train_size
        )
        self.val_dataloader = self.dataset.get_dataloader(
            MASSIVEDatasetSplitName.VAL, self.batch_size, self.val_size
        )
        self.test_dataloader = self.dataset.get_dataloader(
            MASSIVEDatasetSplitName.TEST, self.batch_size, self.test_size
        )
        self.num_epochs = num_epochs
        self.lr = lr
        self.neptune_run = neptune_run

    def _train(self) -> None:
        raise NotImplementedError("Training not implemented.")

    def _test(self) -> None:
        raise NotImplementedError("Testing not implemented.")

    def maybe_save_experiment_params_to_neptune_run(self) -> None:
        if self.neptune_run:
            self.neptune_run["parameters"] = {
                "optimizer_class": str(self.optimizer_class),
                "loss_class": str(self.loss_class),
                "num_epochs": self.num_epochs,
                "lr": self.lr,
                "test_size": self.test_size,
                "val_size": self.val_size,
                "train_size": self.train_size,
                "name": self.name,
                "batch_size": self.batch_size,
                "num_of_layers": self.num_of_layers,
                "train_base_model": self.train_base_model,
                "dropout": self.dropout,
            }

    def maybe_safe_classsidier_layer_state_dict_to_neptune_run(self) -> None:
        raise NotImplementedError("Saving classifier layer not implemented.")

    def run(self) -> None:
        try:
            self.maybe_save_experiment_params_to_neptune_run()
            print(f'Running experiment "{self.name}".')
            self._train()
            print(f'Experiment "{self.name}" finished training.')
            print(f'Running experiment "{self.name}" testing.')
            self._test()
            print(f'Experiment "{self.name}" finished.')
            self.maybe_safe_classsidier_layer_state_dict_to_neptune_run()
            if self.neptune_run:
                self.neptune_run.stop()
        except Exception:
            print("Gracefully stopping neptune run.")
            if self.neptune_run:
                self.neptune_run.stop()
