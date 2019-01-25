#!/bin/bash
# This script should be only executed in docker.
# Run minigo... stop when it converges.
set -e

SEED=$1
shift 1

BASE_DIR=/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/minigo
#BASE_DIR=/research/results/minigo/final/
mkdir -p $BASE_DIR
cd $(dirname $0)/minigo
#cd /research/reinforcement/minigo
#bash ./loop_main.sh params/final.json $SEED
set -x
bash ./loop_main.sh params/james.json $SEED "$@"
