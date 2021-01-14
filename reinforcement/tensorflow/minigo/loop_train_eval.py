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

import glob

from utils import timer
from tensorflow import gfile
import tensorflow as tf
import logging

import goparams
import predict_games
import rlscope.api as rlscope

import qmeas

from mlperf_compliance import mlperf_log

# Pull in environment variables. Run `source ./cluster/common` to set these.
#BUCKET_NAME = os.environ['BUCKET_NAME']

#BASE_DIR = "gs://{}".format(BUCKET_NAME)
BASE_DIR = goparams.BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SELFPLAY_DIR = os.path.join(BASE_DIR, 'data/selfplay')
BURY_DIR = os.path.join(BASE_DIR, 'bury_models')
BURY_SELFPLAY_DIR = os.path.join(BASE_DIR, 'bury_selfplay')
HOLDOUT_DIR = os.path.join(BASE_DIR, 'data/holdout')
SGF_DIR = os.path.join(BASE_DIR, 'sgf')
TRAINING_CHUNK_DIR = os.path.join(BASE_DIR, 'data', 'training_chunks')

ESTIMATOR_WORKING_DIR = os.path.join(BASE_DIR, 'estimator_working_dir')

# How many games before the selfplay workers will stop trying to play more.
MAX_GAMES_PER_GENERATION = goparams.MAX_GAMES_PER_GENERATION

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


def evaluate(prev_model, cur_model, readouts=200, verbose=1, resign_threshold=0.95):
    ''' returns True if cur model should be used in future games '''
    prev_model_save_path = os.path.join(MODELS_DIR, prev_model)
    cur_model_save_path = os.path.join(MODELS_DIR, cur_model)
    game_output_dir = os.path.join(SELFPLAY_DIR, cur_model)
    game_holdout_dir = os.path.join(HOLDOUT_DIR, cur_model)
    sgf_dir = os.path.join(SGF_DIR, cur_model)
    cur_win_pct = main.evaluate_evenly(prev_model_save_path, cur_model_save_path, game_output_dir, readouts=readouts, games=goparams.EVAL_GAMES_PER_SIDE)

    print('Evalute Win Pct = ', cur_win_pct)

    qmeas.record('evaluate_win_pct', cur_win_pct)
    keep = False
    if cur_win_pct >= goparams.EVAL_WIN_PCT_FOR_NEW_MODEL:
      qmeas.record('evaluate_choice', 'new')
      keep = True
    else:
      qmeas.record('evaluate_choice', 'old')
      keep = False
    qmeas.record('eval_summary', {'win_pct': cur_win_pct, 'model': cur_model, 'keep': keep})
    return keep


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
    return new_model_name


def bury_latest_model():
  main._ensure_dir_exists(BURY_DIR)
  main._ensure_dir_exists(BURY_SELFPLAY_DIR)
  model_num, model_name = get_latest_model()
  save_file = os.path.join(MODELS_DIR, model_name)
  cmd = 'mv {}* {}/'.format(save_file, BURY_DIR)
  # delete any selfplay games from that model too
  print('Bury CMD: ', cmd)
  if os.system(cmd) != 0:
    raise Exception('Failed to bury model: ' + cmd)
  cmd = 'mv {}* {}/'.format(os.path.join(SELFPLAY_DIR, model_name), BURY_SELFPLAY_DIR)
  # delete any selfplay games from that model too
  print('Bury Games CMD: ', cmd)
  if os.system(cmd) != 0:
    raise Exception('Failed to bury model: ' + cmd)

  prev_num, prev_model_name = get_latest_model()
  prev_save_file = os.path.join(MODELS_DIR, prev_model_name)

  suffixes = ['.data-00000-of-00001', '.index', '.meta']
  new_name = '{:06d}-continue'.format(model_num)
  new_save_file = os.path.join(MODELS_DIR, new_name)

  for suffix in suffixes:
    cmd = 'cp {} {}'.format(prev_save_file + suffix, new_save_file + suffix)
    print('DBUG ', cmd)
    if os.system(cmd) != 0:
      raise Exception('Failed to copy: ' + cmd)


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


def rl_loop(generation):
    """Run the reinforcement learning loop

    This tries to create a realistic way to run the reinforcement learning with
    all default parameters.
    """
    # RL-Scope: catch-all operation
    with rlscope.prof.operation('train'):
        if goparams.DUMMY_MODEL:
            # monkeypatch the hyperparams so that we get a quickly executing network.
            dual_net.get_default_hyperparams = lambda **kwargs: {
                'k': 8, 'fc_width': 16, 'num_shared_layers': 1, 'l2_strength': 1e-4, 'momentum': 0.9}

            dual_net.TRAIN_BATCH_SIZE = 16
            dual_net.EXAMPLES_PER_GENERATION = 64

            #monkeypatch the shuffle buffer size so we don't spin forever shuffling up positions.
            preprocessing.SHUFFLE_BUFFER_SIZE = 1000

        qmeas.stop_time('selfplay_wait')
        print("Gathering game output...")
        gather()

        print("Training on gathered game data...")
        # import ipdb; ipdb.set_trace()
        _, model_name = get_latest_model()
        # JAMES NOTE: this runs very fast.
        new_model = train()

    if goparams.EVALUATE_PUZZLES:
      # JAMES NOTE: this code evaluates the newly trained model to see whether it
      # reaches the "target accuracy" as defined by MLPerf.
      # If the target accuracy is reached, the minigo benchmark "stops the clock" and
      # records that as its total training time.
      # We DON'T include this in our profiling, since its a termination condition that
      # is specific to MLPerf.
      #
      # The "AlphaGo Zero" paper states that it trained for "700,000" timesteps.
      # It's unclear what terminating condition they used during training.
      # Presumably, they can only evaluate "training progress" by playing games against
      # other pre-trained known-to-be-good models.  For example, playing against the original
      # pre-trained AlphaGo, or a non-neural network agent.  The "test" performance happens when
      # the model finally plays against an expert human.
      # https://deepmind.com/documents/260/alphazero_preprint.pdf
      rlscope.prof.set_phase('evaluate_termination_generation_{g}'.format(
          g=generation,
      ))

      with rlscope.prof.operation('evaluate_term_moves'):

          # JAMES TODO: We'd like to nest 'load_network'/'init_network' inside 'puzzle' here...
          qmeas.start_time('puzzle')
          print("Evaluate puzzles")
          # import ipdb; ipdb.set_trace()
          new_model_path = os.path.join(MODELS_DIR, new_model)
          # JAMES NOTE: pretrained puzzle moves? What are these?
          sgf_files = [
            './benchmark_sgf/9x9_pro_YKSH.sgf',
            './benchmark_sgf/9x9_pro_IYMD.sgf',
            './benchmark_sgf/9x9_pro_YSIY.sgf',
            './benchmark_sgf/9x9_pro_IYHN.sgf',
          ]
          result, total_pct = predict_games.report_for_puzzles(new_model_path, sgf_files, 2, tries_per_move=1)
          print('accuracy = ', total_pct)
          mlperf_log.minigo_print(key=mlperf_log.EVAL_ACCURACY,
                                  value={"epoch": generation, "value": total_pct})
          mlperf_log.minigo_print(key=mlperf_log.EVAL_TARGET,
                                  value=goparams.TERMINATION_ACCURACY)


          qmeas.record('puzzle_total', total_pct)
          qmeas.record('puzzle_result', repr(result))
          qmeas.record('puzzle_summary', {'results': repr(result), 'total_pct': total_pct, 'model': new_model})
          qmeas._flush()
          with open(os.path.join(BASE_DIR, new_model + '-puzzles.txt'), 'w') as f:
            f.write(repr(result))
            f.write('\n' + str(total_pct) + '\n')
          qmeas.stop_time('puzzle')
          if total_pct >= goparams.TERMINATION_ACCURACY:
            print('Reaching termination accuracy; ', goparams.TERMINATION_ACCURACY)

            mlperf_log.minigo_print(key=mlperf_log.RUN_STOP,
                                    value={"success": True})

            with open('TERMINATE_FLAG', 'w') as f:
              f.write(repr(result))
              f.write('\n' + str(total_pct) + '\n')

    # JAMES TODO: with rlscope.prof.use_num_calls(3000):
    # TODO: appears to be running multiple evaluations? (expect 3 games, saw more.)
    if goparams.EVALUATE_MODELS:
      # JAMES NOTE: Evaluate whether the newly trained candidate model out-performs the model from
      # "last generation" (i.e. win percentage > 55%).
      # If it DOESN'T, we recollect data using the OLD model.

      rlscope.prof.set_phase('evaluate_candidate_model_generation_{g}'.format(
          g=generation,
      ))

      with rlscope.prof.operation('evaluate_candidates'):

          # JAMES NOTE: Use the same number of "readouts" as during self-play.
          # NOTE: "readouts" = # of MCTS nodes that get "expanded" BEFORE choosing a move.
          # So expects increasing readouts by a factor of 10 to simply multiply subplot-phase
          # runtime by a factor of 10, with the same CPU/GPU utilization characteristics.

          readouts = goparams.SP_READOUTS
          print("> Evaluate with readouts={readouts}".format(
              readouts=readouts))
          if not evaluate(model_name, new_model, readouts=readouts):
            bury_latest_model()

def main_func():
    logging.info(("> MINIGO CMD:\n"
                  "  $ {cmd}"
                  ).format(cmd=' '.join(sys.argv)))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--generation", type=int, help="Go generation")
    # parser.add_argument("base_dir", type=int, help="generation of self-play/train-eval")
    # parser.add_argument("worker_id", type=int, help="Worker id")
    rlscope.add_rlscope_arguments(parser)
    args = parser.parse_args()
    rlscope.handle_rlscope_args(parser, args, reports_progress=False)

    phase_name = 'sgd_updates_generation_{g}'.format(
        g=args.generation,
    )
    process_name = "loop_train_eval_generation_{g}".format(
        g=args.generation,
    )
    with rlscope.prof.profile(process_name=process_name, phase_name=phase_name, handle_utilization_sampler=False):
        #tf.logging.set_verbosity(tf.logging.INFO)
        seed = args.seed
        generation = args.generation
        print('Setting random seed, generation = ', seed, generation)
        seed = hash(seed) + generation
        print("training seed: ", seed)
        random.seed(seed)
        tf.set_random_seed(seed)
        numpy.random.seed(seed)

        qmeas.start(os.path.join(BASE_DIR, 'stats'))
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
        rl_loop(args.generation)
        qmeas.end()
        mlperf_log.minigo_print(key=mlperf_log.EVAL_STOP, value=generation)

if __name__ == '__main__':
    main_func()
