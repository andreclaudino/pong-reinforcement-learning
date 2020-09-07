from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer


def create_replay_buffer(agent: DdqnAgent, environment: TFEnvironment, max_lenght: int) -> TFUniformReplayBuffer:
    return TFUniformReplayBuffer(agent.collect_data_spec, environment.batch_size, max_length=max_lenght)
