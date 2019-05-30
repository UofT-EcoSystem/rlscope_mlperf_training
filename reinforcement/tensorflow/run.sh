#!/bin/bash
# This script should be only executed in docker.
# Run minigo... stop when it converges.
set -e

SEED=$1
shift 1

cd $(dirname $0)/minigo
#cd /research/reinforcement/minigo
#bash ./loop_main.sh params/final.json $SEED
set -x

if [ "$GOPARAMS" = "" ]; then
    #export GOPARAMS="params/james.json"
    echo "ERROR: You must export GOPARAMS=<full-path>.json"
    exit 1
fi
if [ ! -f "$GOPARAMS" ]; then
    echo "ERROR: No such file @ GOPARAMS = $GOPARAMS"
    exit 1
fi


READ_JSON_PY="read_json.py"
echo "BASE_DIR = $BASE_DIR"
base_dir="$(python3 $READ_JSON_PY $GOPARAMS --attr BASE_DIR --allow-env)"
mkdir -p $base_dir

bash ./loop_main.sh $GOPARAMS $SEED "$@"
