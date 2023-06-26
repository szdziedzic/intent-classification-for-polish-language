import argparse
import os
from enum import Enum
from herbert import HerbertExperiment
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from translator import Translator
from massive_dataset import MASSIVEDataset, MASSIVEDatasetSplitName

parser = argparse.ArgumentParser()

parser.add_argument("--test_size", help="test size", default=None)
parser.add_argument("--val_size", help="validation size", default=None)
parser.add_argument("--train_size", help="train size", default=None)
parser.add_argument("--batch_size", help="batch size", default=128)


if __name__ == "__main__":

    args = parser.parse_args()
    translator = Translator()
    dataset = MASSIVEDataset()
    translator.massive_pl_to_en(dataset.get_dataloader(MASSIVEDatasetSplitName.TRAIN, int(args.batch_size), args.train_size), "translated_train.csv")
    translator.massive_pl_to_en(dataset.get_dataloader(MASSIVEDatasetSplitName.VAL, int(args.batch_size), args.val_size), "translated_val.csv")
    translator.massive_pl_to_en(dataset.get_dataloader(MASSIVEDatasetSplitName.TEST, int(args.batch_size), args.test_size), "translated_test.csv")