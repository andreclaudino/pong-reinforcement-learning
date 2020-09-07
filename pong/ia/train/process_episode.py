from typing import Dict

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_py_policy import TFPyPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory


def collect_episode_data(environment: TFEnvironment, policy: TFPyPolicy, replay_buffer,
                         repeats: int = 1, phase="Training") -> Dict:
    """
    Executa o treinamento até terminar o episódio e gera uma relação de estados e recompensas por frete
    :param environment: Ambiente com o qual interagir para coleta de dados
    :param policy: Política usada para a coleta de dados
    :param replay_buffer: Buffer onde os dados coletados serão armazenados
    :param repeats: number of episodes to process before return
    """
    print(f"Phase is {phase}")
    time_step = environment.reset()

    for _ in range(repeats):
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            collect_step(environment, policy, replay_buffer)

    return environment.get_info()


def collect_step(environment: TFEnvironment, policy: TFPyPolicy, replay_buffer: TFUniformReplayBuffer):
    """
    Coleta uma iteração com o ambiente e devolve o resultado m
    :param environment:
    :param policy:
    :param replay_buffer:
    :return:
    """
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

