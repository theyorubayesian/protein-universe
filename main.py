"""
Written by: Akintunde 'theyorubayesian' Oladipo
14/Nov/2021
"""
import argparse
import logging
import os

import neptune.new as neptune
import torch
from dotenv import load_dotenv
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from universe.constants import VOCAB_SIZE
from universe.dataset import create_dataloader
from universe.models import BiLSTM
from universe.train import train_lstm

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # ----
    # Data
    # ----
    parser.add_argument("--train_data", default="data/processed/300/train.csv")
    parser.add_argument("--val_data", default="data/processed/300/dev.csv")
    parser.add_argument("--test_data", default="data/processed/300/test.csv")
    parser.add_argument("--overwrite_data_cache", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=300)
    
    # -----
    # Model
    # -----
    parser.add_argument("--model_type")
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument("--lstm_hidden_size", type=int, default=128)
    parser.add_argument("--lstm_embedding_dim", type=int, default=128)

    # --------
    # Training
    # --------
    parser.add_argument("--lstm_learning_rate", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--logging_interval", type=int, default=500)
    parser.add_argument("--output_dir", default="outputs")

    args = parser.parse_args()
    args_copy = vars(args)
    
    os.makedirs(args.output_dir, exist_ok=True)

    for a in args_copy:
        logger.info(f"{a}:  {args_copy[a]}")
        
    logging_client = neptune.init(
        project=f"{os.getenv('NEPTUNE_USERNAME')}/{os.getenv('NEPTUNE_PROJECT_NAME')}",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        mode="async",
        source_files=[]
    )
    logging_client["parameters"] = vars(args)
    
    train_dataloader = create_dataloader(
        args.train_data, 
        args.overwrite_data_cache, 
        args.num_classes, 
        "train",
        args.train_batch_size,
        shuffle=True
        )

    val_dataloder = create_dataloader(
        args.val_data,
        args.overwrite_data_cache, 
        args.num_classes, 
        "val",
        args.val_batch_size,
        shuffle=False
    )

    model = BiLSTM(
        hidden_size=args.lstm_hidden_size, 
        num_layers=args.lstm_num_layers, 
        embedding_size=args.lstm_embedding_dim, 
        num_classes=args.num_classes,
        vocab_size=VOCAB_SIZE)

    optimizer = Adam(model.parameters(), lr=args.lstm_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2)

    args.device = torch.device("cuda:0" if (not args.no_cuda) and (torch.cuda.is_available()) else "cpu")
    if args.no_cuda:
        args.n_gpu = 0

    model.to(args.device)

    criterion = nn.CrossEntropyLoss()

    train_lstm(
        train_dataloader,
        val_dataloder,
        model,
        optimizer,
        scheduler,
        criterion,
        logging_client,
        args.num_epochs,
        args
    )


if __name__ == "__main__":
    main()
