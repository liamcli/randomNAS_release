#!/bin/bash
DATA_DIR=/data
RESULTS_DIR=/results
cd /opt/randomNAS/searchers
python -u random_weight_share.py --benchmark cnn --data $DATA_DIR --epochs 150 --batch_size 64 --grad_clip 1 --init_channels 24 --seed $1 --save_dir $RESULT_DIR/cnn/random/trial$1
