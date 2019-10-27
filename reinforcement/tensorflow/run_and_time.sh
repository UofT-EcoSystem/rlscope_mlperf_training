#/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>


set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


# run benchmark

seed=${1:-1}
if [ $# -ge 1 ]; then
    shift 1
fi

echo "running benchmark with seed $seed"
# The termination quality is set in params/final.json. See RAEDME.md.
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi

set +e
./run.sh $seed "$@"
ret_code=$?
set -e

sleep 3
if [ "$ret_code" != "0" ]; then
    echo "ERROR: run.sh failed with ret=$ret_code"
    exit $ret_code
fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"


# report result
result=$(( $end - $start ))
result_name="reinforcement"


echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
