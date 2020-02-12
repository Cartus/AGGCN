AGGCN for Cross Sentence Tenary Task
==========
  

## Requirements

Our model was trained on GTX 1080 .  

- Python 3 (tested on 3.6.8)

- PyTorch (tested on 1.3.1)

- CUDA (tested on 8.0)

- tqdm

- unzip, wget (for downloading only)

There is no guarantee that the model is the same as we released and reported if you run the code on different environments (including hardware and software). 

## Preparation
We have already put the JSON files under the directory `dataset/ter_mul`.

  
First, download and unzip GloVe vectors:

```
chmod +x download.sh; ./download.sh
```

  
Then prepare vocabulary and initial word vectors with:

```
./build_vocab.sh
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/ter_mul/0/vocab`. Note that we use 5 fold cross validation, so this operation is required for 5 splits.

  

## Training

  

To train the AGGCN model, run:

```
./train_script.sh
```

  

Model checkpoints and logs will be saved to `./saved_models`.
  

For details on the use of other parameters, please refer to `train.py`.


Note that we use 5 fold cross validation, so 5 models should be trained independently for 5 splits.
  

## Evaluation

Our pretrained model is saved under the dir saved_models/01. To run evaluation on the test set, run:

```
./test.sh
```


## Citation

```
@inproceedings{guo2019aggcn,
 author = {Guo, Zhijiang and Zhang, Yan and Lu, Wei},
 booktitle = {Proc. of ACL},
 title = {Attention Guided Graph Convolutional Networks for Relation Extraction},
 year = {2019}
}
```
