import os

import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from pong.ia.train.persistense import create_savers
from pong.ia.train.process_episode import collect_episode_data

REPLAY_BUFFER_NUM_STEPS = 5


def train(agent: DdqnAgent, train_env: TFEnvironment,
          replay_buffer: TFUniformReplayBuffer, num_episodes: int,
          replay_buffer_batch_size: int, save_path: str):

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

    for episode in range(num_episodes):
        tf.print(f"Episódio {episode} iniciado")
        episode_info = collect_episode_data(train_env, agent.collect_policy, replay_buffer)

        experience, unused_info = next(train_iterator)
        train_loss = agent.train(experience).loss
        global_step = agent.train_step_counter.numpy()

        tf.print(f"Episódio {episode} finalizado no passo {episode_info['current_step'][0]}, com custo {train_loss}")

        #TODO: Salvar métricas no tensorboard ou outro cara

        # Salvar política e agente
        checkpointer.save(global_step=global_step)
        saver.save(policy_path)

    tf.print(f"Treinamento finalizado no eposiódio {episode} passo {global_step}")
