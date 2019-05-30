#/bin/bash

RANDOM_SEED=$1
QUALITY=$2
set -e

# Register the model as a source root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MODEL_DIR="/tmp/resnet_imagenet_${RANDOM_SEED}"

#DEBUG=yes
DEBUG=no
_python() {
    if [ "$DEBUG" = 'yes' ]; then
        echo python3 -m ipdb
    else
        echo python3
    fi
}
# Having issues running on logan:
# tensorflow.python.framework.errors_impl.InvalidArgumentError: Invalid device ordinal value (1). Valid range is [0, 0].
#         while setting up XLA_GPU_JIT device number 1
export CUDA_VISIBLE_DEVICES="0"
$(_python) official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn/imagenet/combined/  \
  --model_dir $MODEL_DIR --train_epochs 10000 --stop_threshold $QUALITY --batch_size 64 \
  --version 1 --resnet_size 50 --epochs_between_evals 4

# To run on 8xV100s, instead run:
#python3 official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn/imagenet/combined/ \
#   --model_dir $MODEL_DIR --train_epochs 10000 --stop_threshold $QUALITY --batch_size 1024 \
#   --version 1 --resnet_size 50 --dtype fp16 --num_gpus 8 \
#   --epochs_between_evals 4
