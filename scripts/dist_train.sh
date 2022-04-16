#!/usr/bin/env bash

CUDA=$1
CONFIG=$2
PROC=$(awk -F',' '{print NF}' <<< "$CUDA")
PORT=${PORT:-29501}

DATE=$(date +%Y-%m-%d)
SALT=$(openssl rand -base64 6)
echo "Experiment: ${DATE}_${SALT}"
echo "Devices:    $CUDA"
echo "Num.Proc:   $PROC"
echo "Port:       $PORT"

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$CUDA python -m torch.distributed.run --nproc_per_node=$PROC --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --suffix "${DATE}_${SALT}" ${@:4}
