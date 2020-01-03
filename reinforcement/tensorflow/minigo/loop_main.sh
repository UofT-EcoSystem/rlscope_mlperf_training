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

if [ "$DEBUG" == 'yes' ]; then
    set -x
fi

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
    echo "> Cleanup iml-util-sampler"
    cleanup
}
trap ctrl_c EXIT

if [ "$IML_CONFIG" == "" ]; then
    echo "IML ERROR: Expected \"export IML_CONFIG=...\" to be set to either \"full\" (Full IML instrumentation) or \"uninstrumented\" (uninstrumented runs)"
    exit 1
fi

if [ "$IML_DIRECTORY" == "" ]; then
    echo "IML ERROR: Expected \"export IML_DIRECTORY=...\" to be set to directory to store trace-files, but it wasn't set!"
    exit 1
fi

if [ "$BASE_DIR" == "" ]; then
    echo "IML ERROR: Expected \"export BASE_DIR=...\" to be set from reinforcement/tensorflow/run.sh, but it wasn't set!"
    exit 1
fi
echo "BASE_DIR = $BASE_DIR"
echo "IML_DIRECTORY = $IML_DIRECTORY"

NUM_GENERATIONS="$(python3 $READ_JSON_PY $PARAMS_FILE --attr NUM_GENERATIONS)"
iml-util-sampler "$@" --iml-directory $IML_DIRECTORY --iml-root-pid $$ &
UTIL_SAMPLER_PID=$!

GOPARAMS=$PARAMS_FILE iml-prof --config $IML_CONFIG python3 loop_init.py --iml-directory $IML_DIRECTORY --iml-skip-rm-traces "$@"
# JAMES NOTE: I'm not sure WHY they have a loop here.
# If the test-set accuracy >= TERMINATION_ACCURACY (= 0.40), then
# training will terminate.
#
# If this were to have a hyperparameter, it should be called NUM_GENERATIONS.
for i in $(seq 1 $NUM_GENERATIONS);
do
GOPARAMS=$PARAMS_FILE iml-prof --config $IML_CONFIG python3 loop_selfplay.py --seed $SEED --generation $i --iml-directory $IML_DIRECTORY --iml-skip-rm-traces "$@" 2>&1

GOPARAMS=$PARAMS_FILE iml-prof --config $IML_CONFIG python3 loop_train_eval.py --seed $SEED --generation $i --iml-directory $IML_DIRECTORY --iml-skip-rm-traces "$@" 2>&1



if [ -f $FILE ]; then
   echo "$FILE exists: finished!"
   cat $FILE
   break
else
   echo "$FILE does not exist; looping again."
fi
done
