from typing import Sequence

import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.networks.q_network import QNetwork
from tf_agents.utils import common
from tf_agents.utils.common import element_wise_squared_loss
from tf_agents.environments.tf_environment import TFEnvironment


def create_pong_agent(train_environment: TFEnvironment, dense_layer_sizes: Sequence[int],
                      learning_rate: float) -> (DqnAgent, QNetwork):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    q_net = QNetwork(
        input_tensor_spec=train_environment.observation_spec(),
        action_spec=train_environment.action_spec(),
        fc_layer_params=dense_layer_sizes
    )

    agent = DqnAgent(
        time_step_spec=train_environment.time_step_spec(),
        action_spec=train_environment.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=element_wise_squared_loss,
        train_step_counter=global_step
    )

    agent.initialize()
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    return agent
