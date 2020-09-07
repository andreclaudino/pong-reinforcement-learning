import os

import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from pong.ia.model.policy import create_random_policy
from pong.ia.train.persistense import create_savers
from pong.ia.train.process_episode import collect_episode_data

REPLAY_BUFFER_NUM_STEPS = 2


def train(agent: DdqnAgent, train_env: TFEnvironment,
          replay_buffer: TFUniformReplayBuffer, num_episodes: int,
          replay_buffer_batch_size: int, save_path: str, ramdomize_step: int, validate_step: int):

    train_dataset = replay_buffer.as_dataset(num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                             num_steps=REPLAY_BUFFER_NUM_STEPS,
                                             sample_batch_size=replay_buffer_batch_size)

    train_iterator = iter(train_dataset)

    # Savers
    checkpointer, saver = create_savers(save_path, agent, replay_buffer)
    policy_path = os.path.join(save_path, "saved", "policy")

    checkpointer.initialize_or_restore()

    global_step = 0
    episode = 0
    tf.print("Aguardando conexão do cliente")

    random_policy = create_random_policy(train_env)
    collect_episode_data(train_env, random_policy, replay_buffer, repeats=3, phase="Random")

    for episode in range(num_episodes):
        tf.print(f"Episódio {episode} iniciado")
        # Colect random data
        episode_info = collect_episode_data(train_env, agent.collect_policy, replay_buffer, repeats=1)

        experience, unused_info = next(train_iterator)
        train_loss = agent.train(experience).loss
        global_step = agent.train_step_counter.numpy()

        tf.print(f"Episódio {episode} finalizado, com custo {train_loss} e recompensa {episode_info['reward']}")
        collect_episode_data(train_env, agent.policy, replay_buffer, phase="Inference")

        #TODO: Salvar métricas no tensorboard ou outro cara

        # Salvar política e agente
        checkpointer.save(global_step=global_step)
        saver.save(policy_path)

    tf.print(f"Treinamento finalizado no eposiódio {episode} passo {global_step}")
