#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python3 train.py --id bin0 --seed 0 --data_dir dataset/bin_mul/0 --vocab_dir dataset/bin_mul/0
