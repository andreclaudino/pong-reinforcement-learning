from multiprocessing import Queue
from typing import Any

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from pong.messages import Action, Observable

tf.compat.v1.enable_v2_behavior()

NUMBER_OF_ACTIONS = 3
MAX_MESSAGE_ATTEMPS = 1000
INITIAL_STATE = np.array([0.5, 0.5, 0.5, 0.5, 0., 0.])


class PongEnvironment(py_environment.PyEnvironment):

    def __init__(self, agent_mailbox: Queue, service_mailbox: Queue, discount: float):
        super().__init__()

        self._action_spec = array_spec.BoundedArraySpec(shape=(), minimum=0, maximum=(NUMBER_OF_ACTIONS - 1),
                                                        dtype=np.int32, name='action')
        self._observation_spec = array_spec.ArraySpec(shape=(6,), dtype=np.float, name='observation')

        self._state = 0
        self._episode_ended = False

        self._inbox = agent_mailbox
        self._service_mailbox = service_mailbox
        self.discount = discount

        self._phase = None

    def get_state(self) -> Any:
        return self._state

    def get_info(self) -> Any:
        return dict(reward=self.reward)

    def action_spec(self) -> array_spec.ArraySpec:
        return self._action_spec

    def observation_spec(self) -> array_spec.ArraySpec:
        return self._observation_spec

    def _reset(self):
        self._state = INITIAL_STATE
        self._episode_ended = False
        self.reward = 0.0
        return ts.restart(np.array(self._state, dtype=np.float))

    def _step(self, action):
        if self._episode_ended:
            self._send(Action(value=action, phase=self._phase))
            return self.reset()

        self._send(Action(value=action, phase=self._phase))
        message = self._receive()
        finished = message.finished
        reward = message.reward

        if finished:
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.float), reward)

        observation = np.array(message.observable, dtype=np.float)
        return ts.transition(observation, reward, discount=self.discount)

    def _send(self, message: Action):
        self._service_mailbox.put(message)

    def _receive(self) -> Observable:
        count = 0
        while self._inbox.empty():
            count += 1
            if count >= MAX_MESSAGE_ATTEMPS:
                return Observable(INITIAL_STATE, 0, False)
        return self._inbox.get()

    def set_phase(self, phase):
        self._phase = phase


def create_environment(agent_mailbox, service_mailbox, discount) -> TFPyEnvironment:
    environment = PongEnvironment(agent_mailbox, service_mailbox, discount)
    return TFPyEnvironment(environment)
