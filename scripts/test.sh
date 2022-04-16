#!/bin/env/bash

CUDA=$1
TARGET=$2
TEST_ROOT=$3

if [ $# -eq 4 ]; then
    echo "Training on custom checkpoint: $4"
    CHECKPOINT_NAME=$4
    shift 4
else
    shift 3
    CHECKPOINT_NAME="latest"
fi

CWD=$PWD
CONFIG_FILE=$(find ${TEST_ROOT} -type f -name "*.py")
CHECKPOINT_FILE="${TEST_ROOT}/${CHECKPOINT_NAME}.pth"
OUT_DIR="${TEST_ROOT}/preds/"

echo 'TARGET:' $TARGET
echo 'Devices:' $CUDA
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $OUT_DIR
echo ''

if [[ "${TARGET}" == "eval" ]]; then
    CUDA_VISIBLE_DEVICES=${CUDA} python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU
elif [[ "${TARGET}" == "format" ]]; then
    CUDA_VISIBLE_DEVICES=${CUDA} python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
        --out-dir ${OUT_DIR} --format-only
    echo "Zipping images..."
    cd ${OUT_DIR} && zip -r ${CHECKPOINT_NAME}.zip *.png
    cd ${CWD} && mv ${OUT_DIR}${CHECKPOINT_NAME}.zip ${TEST_ROOT}
    echo "Done!"
else
    echo "Command $TARGET not recognized: use 'eval' or 'format'"
fi
