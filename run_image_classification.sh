#!/usr/bin/env bash
ROOT="$(readlink -f $(dirname $0))"
cd $ROOT/image_classification/tensorflow
# cd $HOME/clone/mlperf_training/reinforcement/tensorflow
IMAGE=`docker build . | tail -n 1 | awk '{print $3}'`
SEED=1
NOW=`date "+%F-%T"`
docker run --runtime=nvidia -t -i $IMAGE "./run_and_time.sh" $SEED | tee benchmark-$NOW.log
