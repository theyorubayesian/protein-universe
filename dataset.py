"""

"""
import glob
import logging
import os
import pickle

import pandas as pd
import torch
from pandas import DataFrame
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from constants import ACID_MAP
from constants import ACID_MAP_INV
from constants import PADDING_VALUE

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_dataset_from_files(split: str, data_dir: str) -> DataFrame:
    """
    Reads and concatenates data from multiple files

    split: train|dev|test
    data_dir: Directory containing sub-directories (train/dev/test)

    returns: Dataframe 
    """
    datasets = glob.glob(f"{data_dir}/{split}/*")
    df = pd.concat(map(pd.read_csv, datasets))
    return df


class PfamDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, 
        overwrite_cache: bool = False, 
        cache_dir: str = "data/cache", 
        split_name: str = ''
    ):
        os.makedirs(cache_dir, exist_ok=True)
        cached_features_file = os.path.join(cache_dir, f"cached_dataset_{split_name}")

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file: {cached_features_file}")
            with open(cached_features_file, "rb") as f:
                cache = pickle.load(f)
                self._sequences = cache["sequences"]
                self._labels = cache["labels"]
        else:
            logger.info(f"Creating features from dataframe. Features will be cached to {cached_features_file}")
            self._sequences = (
                df.sequence
                .apply(
                    lambda x: list(map(lambda x: ACID_MAP.get(x, 0), list(x)))
                )
            )
            labels = {v: k for k, v in enumerate(df.family_accession.unique())}
            self._labels = df.family_accession.map(labels).tolist()

            with open(cached_features_file, "wb") as f:
                pickle.dump(
                    {"sequences": self._sequences, "labels": self._labels},
                    f, protocol=pickle.HIGHEST_PROTOCOL
                )

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self._sequences[idx],
            "label": self._labels[idx]
        }

    @staticmethod
    def collate(features):
        # TODO: Position IDs?
        input_ids = pad_sequence(
            [torch.tensor(f["sequence"], dtype=torch.long) for f in features],
            batch_first=True, padding_value=PADDING_VALUE
            )
        labels = torch.tensor([f["label"] for f in features])

        return input_ids, labels


class LengthSampler(Sampler):
    pass
