# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper scripts to ensure that main.py commands are called correctly."""
import argh
import argparse
import cloud_logging
import logging
import os
import main
import shipname
import sys
import time
import shutil
import dual_net
import preprocessing
import numpy
import random

from utils import timer
from tensorflow import gfile
import tensorflow as tf
import logging

import goparams
import rlscope.api as rlscope

from loop_selfplay import SelfplayGlobals

import qmeas
import multiprocessing

# Pull in environment variables. Run `source ./cluster/common` to set these.
#BUCKET_NAME = os.environ['BUCKET_NAME']

#BASE_DIR = "gs://{}".format(BUCKET_NAME)
#BASE_DIR = goparams.BASE_DIR
# BASE_DIR = sys.argv[1]

BASE_DIR = None
MODELS_DIR = None
SELFPLAY_DIR = None
HOLDOUT_DIR = None
SGF_DIR = None
TRAINING_CHUNK_DIR = None
ESTIMATOR_WORKING_DIR = None
HOLDOUT_PCT = None

def init_globals(base_dir):
    global \
    BASE_DIR, \
    MODELS_DIR, \
    SELFPLAY_DIR, \
    HOLDOUT_DIR, \
    SGF_DIR, \
    TRAINING_CHUNK_DIR, \
    ESTIMATOR_WORKING_DIR, \
    HOLDOUT_PCT \

    BASE_DIR = base_dir

    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    SELFPLAY_DIR = os.path.join(BASE_DIR, 'data/selfplay')
    HOLDOUT_DIR = os.path.join(BASE_DIR, 'data/holdout')
    SGF_DIR = os.path.join(BASE_DIR, 'sgf')
    TRAINING_CHUNK_DIR = os.path.join(BASE_DIR, 'data', 'training_chunks')

    ESTIMATOR_WORKING_DIR = os.path.join(BASE_DIR, 'estimator_working_dir')

    # What percent of games to holdout from training per generation

    HOLDOUT_PCT = goparams.HOLDOUT_PCT

def print_flags():
    flags = {
        #'BUCKET_NAME': BUCKET_NAME,
        'BASE_DIR': BASE_DIR,
        'MODELS_DIR': MODELS_DIR,
        'SELFPLAY_DIR': SELFPLAY_DIR,
        'HOLDOUT_DIR': HOLDOUT_DIR,
        'SGF_DIR': SGF_DIR,
        'TRAINING_CHUNK_DIR': TRAINING_CHUNK_DIR,
        'ESTIMATOR_WORKING_DIR': ESTIMATOR_WORKING_DIR,
    }
    print("Computed variables are:")
    print('\n'.join('--{}={}'.format(flag, value)
                    for flag, value in flags.items()))


def get_models():
    """Finds all models, returning a list of model number and names
    sorted increasing.

    Returns: [(13, 000013-modelname), (17, 000017-modelname), ...etc]
    """
    all_models = gfile.Glob(os.path.join(MODELS_DIR, '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = sorted([
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames])
    return model_numbers_names


def get_latest_model():
    """Finds the latest model, returning its model number and name

    Returns: (17, 000017-modelname)
    """
    models = get_models()
    if len(models) == 0:
        models = [(0, '000000-bootstrap')]
    return models[-1]


def get_model(model_num):
    models = {k: v for k, v in get_models()}
    if not model_num in models:
        raise ValueError("Model {} not found!".format(model_num))
    return models[model_num]


def game_counts(n_back=20):
    """Prints statistics for the most recent n_back models"""
    all_models = gfile.Glob(os.path.join(MODELS_DIR, '*.meta'))
    model_filenames = sorted([os.path.basename(m).split('.')[0]
                              for m in all_models], reverse=True)
    for m in model_filenames[:n_back]:
        games = gfile.Glob(os.path.join(SELFPLAY_DIR, m, '*.zz'))
        print(m, len(games))


def bootstrap():
    bootstrap_name = shipname.generate(0)
    bootstrap_model_path = os.path.join(MODELS_DIR, bootstrap_name)
    print("Bootstrapping with working dir {}\n Model 0 exported to {}".format(
        ESTIMATOR_WORKING_DIR, bootstrap_model_path))
    main.bootstrap(ESTIMATOR_WORKING_DIR, bootstrap_model_path)


def selfplay(model_name, readouts=goparams.SP_READOUTS, verbose=1, resign_threshold=0.95):
    print("Playing a game with model {}".format(model_name))
    model_save_path = os.path.join(MODELS_DIR, model_name)
    game_output_dir = os.path.join(SELFPLAY_DIR, model_name)
    game_holdout_dir = os.path.join(HOLDOUT_DIR, model_name)
    sgf_dir = os.path.join(SGF_DIR, model_name)
    main.selfplay(
        load_file=model_save_path,
        output_dir=game_output_dir,
        holdout_dir=game_holdout_dir,
        output_sgf=sgf_dir,
        readouts=readouts,
        holdout_pct=HOLDOUT_PCT,
        resign_threshold=resign_threshold,
        verbose=verbose,
    )


def selfplay_cache_model(network, model_name, readouts=goparams.SP_READOUTS, verbose=1, resign_threshold=0.95):
    print("Playing a game with model {}".format(model_name))
    game_output_dir = os.path.join(SELFPLAY_DIR, model_name)
    game_holdout_dir = os.path.join(HOLDOUT_DIR, model_name)
    sgf_dir = os.path.join(SGF_DIR, model_name)
    main.selfplay_cache_model(
        network=network,
        output_dir=game_output_dir,
        holdout_dir=game_holdout_dir,
        output_sgf=sgf_dir,
        readouts=readouts,
        holdout_pct=HOLDOUT_PCT,
        resign_threshold=resign_threshold,
        verbose=verbose,
    )
    # Increment number of selfplay games played.
    selfplay_globals = SelfplayGlobals(SELFPLAY_DIR, model_name)
    num_games_played = selfplay_globals.increment_games_played()
    print("Played {n} games so far with model {model}".format(
        n=num_games_played,
        model=model_name))


def gather():
    print("Gathering game output...")
    main.gather(input_directory=SELFPLAY_DIR,
                output_directory=TRAINING_CHUNK_DIR)


def train():
    model_num, model_name = get_latest_model()
    print("Training on gathered game data, initializing from {}".format(model_name))
    new_model_name = shipname.generate(model_num + 1)
    print("New model will be {}".format(new_model_name))
    load_file = os.path.join(MODELS_DIR, model_name)
    save_file = os.path.join(MODELS_DIR, new_model_name)
    #try:
    main.train(ESTIMATOR_WORKING_DIR, TRAINING_CHUNK_DIR, save_file,
               generation_num=model_num + 1)
    #except:
    #    print("Got an error training, muddling on...")
    #    logging.exception("Train error")


def validate(model_num=None, validate_name=None):
    """ Runs validate on the directories up to the most recent model, or up to
    (but not including) the model specified by `model_num`
    """
    if model_num is None:
        model_num, model_name = get_latest_model()
    else:
        model_num = int(model_num)
        model_name = get_model(model_num)

    # Model N was trained on games up through model N-2, so the validation set
    # should only be for models through N-2 as well, thus the (model_num - 1)
    # term.
    models = list(
        filter(lambda num_name: num_name[0] < (model_num - 1), get_models()))
    # Run on the most recent 50 generations,
    # TODO(brianklee): make this hyperparameter dependency explicit/not hardcoded
    holdout_dirs = [os.path.join(HOLDOUT_DIR, pair[1])
                    for pair in models[-50:]]

    main.validate(ESTIMATOR_WORKING_DIR, *holdout_dirs,
                  checkpoint_name=os.path.join(MODELS_DIR, model_name),
                  validate_name=validate_name)


def echo():
    pass  # Flags are echo'd in the ifmain block below.


def selfplay_hook(args):
  selfplay(**args)


def selfplay_laod_model(model_name):
    load_file = os.path.join(MODELS_DIR, model_name)
    network = dual_net.DualNetwork(load_file)
    return network


def rl_loop():
    """Run the reinforcement learning loop

    This tries to create a realistic way to run the reinforcement learning with
    all default parameters.
    """

    # RL-Scope: catch-all operation
    with rlscope.prof.operation('selfplay_worker'):

        if goparams.DUMMY_MODEL:
            # monkeypatch the hyperparams so that we get a quickly executing network.
            dual_net.get_default_hyperparams = lambda **kwargs: {
                'k': 8, 'fc_width': 16, 'num_shared_layers': 1, 'l2_strength': 1e-4, 'momentum': 0.9}

            dual_net.TRAIN_BATCH_SIZE = 16
            dual_net.EXAMPLES_PER_GENERATION = 64

            #monkeypatch the shuffle buffer size so we don't spin forever shuffling up positions.
            preprocessing.SHUFFLE_BUFFER_SIZE = 1000

        _, model_name = get_latest_model()
        selfplay_globals = SelfplayGlobals(SELFPLAY_DIR, model_name)
        network = selfplay_laod_model(model_name)

        while True:
            should_play_game = selfplay_globals.maybe_increment_games_to_be_played(max_limit=goparams.MAX_GAMES_PER_GENERATION)
            if not should_play_game:
                break
            selfplay_cache_model(network, model_name)
        games = selfplay_globals.count_games_to_be_played()

        print('Stopping selfplay after finding {} games to be played.'.format(games))


if __name__ == '__main__':
    logging.info(("> MINIGO CMD:\n"
                  "  $ {cmd}"
                  ).format(cmd=' '.join(sys.argv)))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # GOPARAMS=params/james.json python3 selfplay_worker.py
    # /mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/minigo 2
    # --worker-id 0 loop_selfplay.py --rlscope-num-traces 10
    # --rlscope-start-measuring-call 1 --rlscope-bench-name NoBenchName
    # --rlscope-num-calls 1000
    parser.add_argument("--base-dir", help="Iteration of self-play/train-eval")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--generation", type=int, help="Go generation")
    parser.add_argument("--worker-id", type=int, required=True, help="Worker id")
    rlscope.add_rlscope_arguments(parser)
    args = parser.parse_args()
    rlscope.handle_rlscope_args(parser, args, reports_progress=False)

    init_globals(args.base_dir)

    process_name = "selfplay_worker_{i}_generation_{g}".format(
        i=args.worker_id,
        g=args.generation,
    )
    with rlscope.prof.profile(process_name=process_name, phase_name=process_name, handle_utilization_sampler=False):

        #tf.logging.set_verbosity(tf.logging.INFO)
        seed = args.seed
        print('Self play worker: setting random seed = ', seed)
        random.seed(seed)
        tf.set_random_seed(seed)
        numpy.random.seed(seed)

        if goparams.TENSORFLOW_LOGGING:
            # get TF logger
            log = logging.getLogger('tensorflow')
            log.setLevel(logging.DEBUG)

            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # create file handler which logs even debug messages
            fh = logging.FileHandler('tensorflow.log')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            log.addHandler(fh)
        rl_loop()

