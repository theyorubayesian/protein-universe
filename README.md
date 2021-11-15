# protein-universe
This project explores how deep learning models can be built to classify protein sequences. As of 14th November 2021, only BiLSTM has been implemented. 

## Setup and Installation
1. Install all the required packages
```
pip install requirements.txt
```
2. Install the version of Pytorch compatible with your CUDA version. Note that this project was only tested on Pytorch 1.10.
3. Experiment data is logged to Neptune. You need to set the environment variables listed in `.envexample`.

## Data
We use the Pfam Protein Dataset. See [Deep Learning Classifies the Protein Universe](https://research.google/pubs/pub48390/).
The dataset is available on [Kaggle](https://www.kaggle.com/googleai/pfam-seed-random-split). To download and setup:
```
bash setup.sh
```

## Training
Model definition, training and evaluation scripts are contained in the `universe` module. To train a model
```
python main.py --data_dir data --overwrite_data_cache --num_classes 300 \
--lstm_num_layers 3 --num_epochs 10 --logging_interval 300
```
Other possible arguments can be checked in the `main.py`

# Evaluation
To evaluate a trained model
```
python evaluate.py --model_path outputs/val-checkpoint-3287 --test_data data/processes/300/test.csv --batch_size 128
```

