#!/bin/bash
#
# We achieve parallelism through multiprocessing for minigo.
# This technique is rather crude, but gives the necessary
# speedup to run the benchmark in a useful length of time.

set -e

PARAMS_FILE=$1
SEED=$2
shift 2

FILE="TERMINATE_FLAG"
rm -f $FILE

set -x

GOPARAMS=$PARAMS_FILE python3 -m ipdb loop_init.py "$@"
#for i in {0..1000};
#do
#GOPARAMS=$PARAMS_FILE python3 loop_selfplay.py $SEED $i "$@" 2>&1
#
#GOPARAMS=$PARAMS_FILE python3 loop_train_eval.py $SEED $i "$@" 2>&1
#
#
#
#if [ -f $FILE ]; then
#   echo "$FILE exists: finished!"
#   cat $FILE
#   break
#else
#   echo "$FILE does not exist; looping again."
#fi
#done
