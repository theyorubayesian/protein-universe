import glob

import pandas as pd
from pandas import DataFrame


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

