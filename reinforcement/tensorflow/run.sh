#!/bin/bash
# This script should be only executed in docker.
# Run minigo... stop when it converges.
set -e

SEED=$1
shift 1

cd $(dirname $0)/minigo
#cd /research/reinforcement/minigo
#bash ./loop_main.sh params/final.json $SEED
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi

if [ "$GOPARAMS" = "" ]; then
    #export GOPARAMS="params/james.json"
    echo "ERROR: You must export GOPARAMS=<full-path>.json"
    exit 1
fi
if [ ! -f "$GOPARAMS" ]; then
    echo "ERROR: No such file @ GOPARAMS = $GOPARAMS"
    exit 1
fi


#READ_JSON_PY="read_json.py"
#echo "BASE_DIR = $BASE_DIR"
#base_dir="$(python3 $READ_JSON_PY $GOPARAMS --attr BASE_DIR --allow-env)"

if [ "$IML_DIRECTORY" == "" ]; then
    echo "IML ERROR: Expected \"export IML_DIRECTORY=...\" to be set to directory to store trace-files, but it wasn't set!"
    exit 1
fi
export BASE_DIR=$IML_DIRECTORY/minigo_base_dir

if [ "$IML_CONFIG" == "" ]; then
    echo "IML ERROR: Expected \"export IML_CONFIG=...\" to be set to either \"full\" (Full IML instrumentation) or \"uninstrumented\" (uninstrumented runs)"
    exit 1
fi

if [ -d "$IML_DIRECTORY" ]; then
    echo "> Detected results from previous IML run @ $IML_DIRECTORY; deleting them:"
    echo "  RM: $IML_DIRECTORY"
    rm -rf $IML_DIRECTORY
fi

if [ -d "$BASE_DIR" ]; then
    echo "> Detected results from previous minigo run @ $BASE_DIR; deleting them:"
    echo "  RM: $BASE_DIR"
    rm -rf $BASE_DIR
fi

mkdir -p $BASE_DIR

bash ./loop_main.sh $GOPARAMS $SEED "$@"
