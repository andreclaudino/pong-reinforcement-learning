import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import numpy as np

tf.compat.v1.enable_v2_behavior()


NUM_ACTIONS = 3


class PongEnv(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()

        self._action_spec =\
            array_spec.ArraySpec(shape=(3,), dtype=np.float, name='action')
        self._observation_spec =\
            array_spec.ArraySpec(shape=(6,), dtype=np.float32, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.array([0, ])
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))


