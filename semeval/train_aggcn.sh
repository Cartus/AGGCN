#!/bin/bash

CUDA_VISIBLE_DEVICES=2  python3 train.py --id 1 --seed 1 --hidden_dim 360 --lr 0.5 --rnn_hidden 300 --num_epoch 150 --pooling max  --mlp_layers 1 --num_layers 2 --pooling_l2 0.002

