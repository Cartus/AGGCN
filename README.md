Attention Guided Graph Convolutional Networks for Relation Extraction
==========

This repo contains the *PyTorch* code for the paper. 

This paper/code introduces the Attention Guided Graph Convolutional graph convolutional networks (AGGCNs) over dependency trees for the large scale sentence-level relation extraction task (TACRED). 

## Requirements

- Python 3 (tested on 3.6.5)
- PyTorch (tested on 0.4.1)
- tqdm
- unzip, wget (for downloading only)

## Preparation

The code requires that you have access to the TACRED dataset (LDC license required). Once you have the TACRED data, please put the JSON files under the directory `dataset/tacred`. 

First, download and unzip GloVe vectors:
```
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

## Training

To train the AGGCN model, run:
```
bash train_caggcn.sh 1
```

Model checkpoints and logs will be saved to `./saved_models/01`.

For details on the use of other parameters, please refer to `train.py`.

## Evaluation

To run evaluation on the test set, run:
```
python eval.py saved_models/01 --dataset test
```

This will use the `best_model.pt` file by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file.

