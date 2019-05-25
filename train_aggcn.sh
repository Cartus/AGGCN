#!/bin/bash

SAVE_ID=$1
CUDA_VISIBLE_DEVICES=4  python train.py --id $SAVE_ID --seed 0 --hidden_dim 300 --lr 0.7 --rnn_hidden 300 --num_epoch 100 --pooling max  --mlp_layers 1 --num_layers 2 --pooling_l2 0.002

