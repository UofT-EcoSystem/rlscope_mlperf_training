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
import shipname
import sys
import time
import shutil
import dual_net
import preprocessing
import subprocess
import pprint
import textwrap

import codecs
import json
import filelock

import glob
from tensorflow import gfile

from utils import timer
import logging

import goparams
import predict_moves

import qmeas

import rlscope.api as rlscope

SEED = None
ITERATION = None

# Pull in environment variables. Run `source ./cluster/common` to set these.
#BUCKET_NAME = os.environ['BUCKET_NAME']

#BASE_DIR = "gs://{}".format(BUCKET_NAME)
BASE_DIR = goparams.BASE_DIR

MODELS_DIR = os.path.join(BASE_DIR, 'models')
SELFPLAY_DIR = os.path.join(BASE_DIR, 'data/selfplay')
BURY_DIR = os.path.join(BASE_DIR, 'bury_models')
HOLDOUT_DIR = os.path.join(BASE_DIR, 'data/holdout')
SGF_DIR = os.path.join(BASE_DIR, 'sgf')
TRAINING_CHUNK_DIR = os.path.join(BASE_DIR, 'data', 'training_chunks')

ESTIMATOR_WORKING_DIR = os.path.join(BASE_DIR, 'estimator_working_dir')

# How many games before the selfplay workers will stop trying to play more.
MAX_GAMES_PER_GENERATION = goparams.MAX_GAMES_PER_GENERATION

# What percent of games to holdout from training per generation

HOLDOUT_PCT = goparams.HOLDOUT_PCT

def as_lock_path(path):
    return "{path}.lock".format(
        path=path,
    )

class SelfplayGlobals:
    """
    globals.json
    {
        # Workers increment this BEFORE they play a game, which informs other workers about the
        # EVENTUAL number of games that will be played.
        # This allows us to avoid "playing too many games".
        games_to_be_played: <Integer>,

        # Games that have been played.
        # NOTE: not strictly required, just nice for debugging progress.
        games_played: <Integer>,
    }
    """
    def __init__(self, selfplay_dir, model_name):
        self.json_path = os.path.join(selfplay_dir, model_name, 'globals.json')
        self.lock_path = as_lock_path(self.json_path)

    def init_state(self):
        # Shouldn't exist yet...
        assert not os.path.exists(self.lock_path)
        lock = filelock.FileLock(self.lock_path)
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        with lock:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            data = {
                'games_to_be_played': 0,
                'games_played': 0,
            }
            self._dump_json(data, self.json_path)

    def count_games_to_be_played(self):
        return self._get_attr('games_to_be_played')

    def count_games_played(self):
        return self._get_attr('games_played')

    def _get_attr(self, attr):
        assert os.path.exists(self.lock_path)
        lock = filelock.FileLock(self.lock_path)
        with lock:
            data = self._load_json(self.json_path)
        return data[attr]

    def increment_games_played(self):
        assert os.path.exists(self.lock_path)
        lock = filelock.FileLock(self.lock_path)
        with lock:
            data = self._load_json(self.json_path)
            data['games_played'] = data['games_played'] + 1
            ret = data['games_played']
            self._dump_json(data, self.json_path)
        return ret

    def maybe_increment_games_to_be_played(self, max_limit):
        """
        PSEUDOCODE:
        lock state:
            if state.games_to_be_played < goparams.MAX_GAMES_PER_GENERATION:
                state.games_to_be_played += 1
                return True
            else:
                return False
        """
        assert os.path.exists(self.lock_path)
        lock = filelock.FileLock(self.lock_path)
        with lock:
            data = self._load_json(self.json_path)
            if data['games_to_be_played'] < max_limit:
                data['games_to_be_played'] = data['games_to_be_played'] + 1
                self._dump_json(data, self.json_path)
                ret = True
            else:
                # Max number of games to play has been reached.
                ret = False
        return ret

    def _dump_json(self, data, json_path):
        with codecs.open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(data, f,
                      sort_keys=True, indent=4,
                      skipkeys=False)

    def _load_json(self, json_path):
        with codecs.open(json_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
        return data


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



def main_(seed, generation):
    """Run the reinforcement learning loop

    This tries to create a realistic way to run the reinforcement learning with
    all default parameters.
    """
    print('Starting self play loop.')

    # JAMES TODO: Add set_phase / end_phase (a phase in the RL case is a "generation":
    # an "iteration" of loop_selfplay.py then loop_train_eval.py).

    qmeas.start_time('selfplay_wait')
    start_t = time.time()

    _, model_name = get_latest_model()

    num_workers = 0

    procs = [
    ]
    def count_live_procs():
      return len(list(filter(lambda proc: proc.poll() is None, procs)))
    def start_worker(i, num_workers):
      #procs.append(subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE))
      worker_seed = hash(hash(SEED) + ITERATION) + num_workers
      # JAMES TODO: forward set_phase to children.
      rlscope_argv, rlscope_env = rlscope.rlscope_argv_and_env(rlscope.prof)
      cmdline_args = []
      cmdline_args.extend(["GOPARAMS={GOPARAMS}".format(
          GOPARAMS=os.environ['GOPARAMS'])])
      logging.info("> goparams.RUN_NVPROF = {RUN_NVPROF}".format(RUN_NVPROF=goparams.RUN_NVPROF))
      if goparams.RUN_NVPROF and i < goparams.RUN_NVPROF_SELFPLAY_WORKERS:
          nvprof_dir = os.path.join(BASE_DIR, "nvprof", "profile.process_%p.nvprof")
          os.makedirs(os.path.dirname(nvprof_dir), exist_ok=True)
          # default=8MB; causes warnings about invalid timestamps from buffer being too small with 16 workers.
          # Double size until error doesn't happen.
          cmdline_args.extend([
              "nvprof",
              "--profile-child-processes",
              "-o", nvprof_dir,
              # "--device-buffer-size", 16,
          ])
      else:
          cmdline_args.extend(["rls-prof", "--config", goparams.RLSCOPE_CONFIG])
      cmdline_args.extend([
          "python3", "selfplay_worker.py",
          "--base-dir", BASE_DIR,
          "--seed", seed,
          "--generation", generation,
          "--worker-id", i,
         ])
      if goparams.RUN_NVPROF:
          cmdline_args.extend(["--rlscope-disable"])
      cmdline_args.extend(rlscope_argv)
      cmd = " ".join([str(opt) for opt in cmdline_args])
      print("> CMDLINE @ worker_{i}:\n  $ {cmd}".format(
          i=i, cmd=cmd))
      print(textwrap.indent(pprint.pformat({
          'env': rlscope_env,
      }), prefix="  "))
      procs.append(subprocess.Popen(cmd, shell=True, env=rlscope_env))

    selfplay_dir = os.path.join(SELFPLAY_DIR, model_name)

    selfplay_globals = SelfplayGlobals(SELFPLAY_DIR, model_name)
    selfplay_globals.init_state()

    print('NUM_PARALLEL_SELFPLAY = {n}'.format(n=goparams.NUM_PARALLEL_SELFPLAY))
    for i in range(goparams.NUM_PARALLEL_SELFPLAY):
      print('Starting Worker...')
      num_workers += 1
      start_worker(i, num_workers)
      time.sleep(1)
    sys.stdout.flush()

    def check_procs():
        failed = False
        for i, proc in enumerate(procs):
            ret = proc.poll()
            if ret is not None and ret != 0:
                failed = True
                print("> Self-play worker failed: worker_id={i}".format(i=i))
        if failed:
            print("> FAILED!".format(i=i))
            sys.exit(1)

    games = selfplay_globals.count_games_to_be_played()
    while games < MAX_GAMES_PER_GENERATION:
        time.sleep(10)
        check_procs()

        games = selfplay_globals.count_games_to_be_played()
        print('Found Games: {}'.format(games))
        print('selfplaying: {:.2f} games/hour'.format(games / ((time.time() - start_t) / 60 / 60) ))
        print('Worker Processes: {}'.format(count_live_procs()))
        sys.stdout.flush()

    print('Done with selfplay loop.')

    print('Waiting for selfplay worker children to exit...')
    for proc in procs:
        proc.wait()
    print('Done waiting for selfplay worker children')

    # Because we use process level parallelism for selfpaying and we don't
    # sync or communicate between processes, there could be too many games
    # played (up to 1 extra game per worker process).
    # This is a rather brutish way to ensure we train on the correct number
    # of games...
    games = selfplay_globals.count_games_played()
    print('There are {} games in the selfplay directory at {}'.format(games, selfplay_dir))
    sys.stdout.flush()

    # There shouldn't be any "extra games"
    assert games == MAX_GAMES_PER_GENERATION

    qmeas.stop_time('selfplay_wait')


if __name__ == '__main__':
    logging.info(("> CHANGE MINIGO CMD:\n"
                  "  $ {cmd}"
                  ).format(cmd=' '.join(sys.argv)))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--generation", type=int, help="Go generation")
    rlscope.add_rlscope_arguments(parser)
    args = parser.parse_args()
    rlscope.handle_rlscope_args(parser, args, reports_progress=False)

    #tf.logging.set_verbosity(tf.logging.INFO)
    qmeas.start(os.path.join(BASE_DIR, 'stats'))

    phase_name = 'selfplay_workers_generation_{g}'.format(
        g=args.generation,
    )
    process_name = 'loop_selfplay_generation_{g}'.format(
        g=args.generation,
    )
    with rlscope.prof.profile(process_name=process_name, phase_name=phase_name, handle_utilization_sampler=False):

        SEED = args.seed
        ITERATION = args.generation

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
        main_(args.seed, args.generation)

    qmeas.end()

