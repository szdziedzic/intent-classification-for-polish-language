from datasets import load_dataset
from torch.utils import data
from enum import Enum
from typing import Union

MASSIVE_DATASET_INTENTS = [
    "datetime_query",
    "iot_hue_lightchange",
    "transport_ticket",
    "takeaway_query",
    "qa_stock",
    "general_greet",
    "recommendation_events",
    "music_dislikeness",
    "iot_wemo_off",
    "cooking_recipe",
    "qa_currency",
    "transport_traffic",
    "general_quirky",
    "weather_query",
    "audio_volume_up",
    "email_addcontact",
    "takeaway_order",
    "email_querycontact",
    "iot_hue_lightup",
    "recommendation_locations",
    "play_audiobook",
    "lists_createoradd",
    "news_query",
    "alarm_query",
    "iot_wemo_on",
    "general_joke",
    "qa_definition",
    "social_query",
    "music_settings",
    "audio_volume_other",
    "calendar_remove",
    "iot_hue_lightdim",
    "calendar_query",
    "email_sendemail",
    "iot_cleaning",
    "audio_volume_down",
    "play_radio",
    "cooking_query",
    "datetime_convert",
    "qa_maths",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "transport_query",
    "music_likeness",
    "email_query",
    "play_music",
    "audio_volume_mute",
    "social_post",
    "alarm_set",
    "qa_factoid",
    "calendar_set",
    "play_game",
    "alarm_remove",
    "lists_remove",
    "transport_taxi",
    "recommendation_movies",
    "iot_coffee",
    "music_query",
    "play_podcasts",
    "lists_query",
]

MASSIVE_DATASET_HUGGINGFACE_ID = "AmazonScience/massive"


class MASSIVEDatasetTorchDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        return example["utt"], example["intent"]


class MASSIVEDatasetSplitName(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class MASSIVEDataset:
    def __init__(self, lang: str = "pl-PL"):
        self.split = {
            MASSIVEDatasetSplitName.TRAIN: load_dataset(
                MASSIVE_DATASET_HUGGINGFACE_ID, lang, split="train"
            ),
            MASSIVEDatasetSplitName.VAL: load_dataset(
                MASSIVE_DATASET_HUGGINGFACE_ID, lang, split="validation"
            ),
            MASSIVEDatasetSplitName.TEST: load_dataset(
                MASSIVE_DATASET_HUGGINGFACE_ID, lang, split="test"
            ),
        }

    def get_dataloader(
        self,
        split_name: MASSIVEDatasetSplitName,
        size: Union[int, None] = None,
    ):
        if size is None:
            return data.DataLoader(
                MASSIVEDatasetTorchDataset(self.split[split_name]),
                batch_size=32,
                shuffle=True,
                num_workers=4,
            )
        return data.DataLoader(
            data.Subset(
                MASSIVEDatasetTorchDataset(self.split[split_name]),
                range(size),
            ),
            batch_size=32,
            shuffle=True,
            num_workers=4,
        )
