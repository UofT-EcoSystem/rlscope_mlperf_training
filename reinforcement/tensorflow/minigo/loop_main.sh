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

READ_JSON_PY="read_json.py"

echo "> CMDLINE: $0 $@"

function cleanup() {
    if [ ! -z "$UTIL_SAMPLER_PID" ]; then
        echo "> Stopping utilization_sampler: pid=$UTIL_SAMPLER_PID"
        kill -TERM $UTIL_SAMPLER_PID
        wait $UTIL_SAMPLER_PID || true
        UTIL_SAMPLER_PID=
    fi
}

UTIL_SAMPLER_PID=
function ctrl_c() {
    echo "> Received SIGINT/SIGTERM"
    cleanup
}
trap ctrl_c SIGINT SIGTERM

#NUM_GENERATIONS="$(python3 $READ_JSON_PY $PARAMS_FILE --attr NUM_GENERATIONS --default 1000)"
NUM_GENERATIONS="$(python3 $READ_JSON_PY $PARAMS_FILE --attr NUM_GENERATIONS)"
echo "BASE_DIR = $BASE_DIR"
base_dir="$(python3 $READ_JSON_PY $PARAMS_FILE --attr BASE_DIR --allow-env)"
python3 -m scripts.utilization_sampler "$@" --iml-directory $base_dir &
UTIL_SAMPLER_PID=$!

GOPARAMS=$PARAMS_FILE python3 loop_init.py "$@"
# JAMES NOTE: I'm not sure WHY they have a loop here.
# If the test-set accuracy >= TERMINATION_ACCURACY (= 0.40), then
# training will terminate.
#
# If this were to have a hyperparameter, it should be called NUM_GENERATIONS.
for i in $(seq 0 $NUM_GENERATIONS);
do
GOPARAMS=$PARAMS_FILE python3 loop_selfplay.py $SEED $i "$@" 2>&1

GOPARAMS=$PARAMS_FILE python3 loop_train_eval.py $SEED $i "$@" 2>&1



if [ -f $FILE ]; then
   echo "$FILE exists: finished!"
   cat $FILE
   break
else
   echo "$FILE does not exist; looping again."
fi
done

cleanup
