#!/bin/bash
DATA_DIR=/data
RESULTS_DIR=/results

source ./data/aws_creds.sh

cd /opt/darts_fork/cnn
git pull

export PYTHONPATH=/opt/darts_fork:/opt/darts_fork/cnn

python train_aws.py --data $DATA_DIR --save RANDOM$1 --arch RANDOM$1 --seed $1 --epochs 600 --save_to_remote --auxiliary --cutout
