#!/bin/bash
DATA_DIR=/data
RESULTS_DIR=/results

./data/aws_creds.sh

cd /opt/randomNAS/searchers
git checkout origin/development 
export PYTHONPATH=/opt/darts_fork:/opt/randomNAS:/opt/darts_fork/cnn
python -u random_weight_share.py --data_dir $DATA_DIR --benchmark cnn --seed $1 --epochs $2 --batch_size 64 --grad_clip 1 --init_channels $3 --save_dir $RESULT_DIR/cnn/random/trial$1 --save_to_remote
