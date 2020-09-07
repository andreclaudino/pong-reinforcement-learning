from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.q_policy import QPolicy
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.random_tf_policy import RandomTFPolicy


def create_policy(env: TFEnvironment, q_network: QNetwork) -> QPolicy:
    return QPolicy(time_step_spec=env.time_step_spec(),
                   action_spec=env.action_spec(),
                   q_network=q_network)


def create_random_policy(env: TFEnvironment):
    return RandomTFPolicy(action_spec=env.action_spec(), time_step_spec=env.time_step_spec())
