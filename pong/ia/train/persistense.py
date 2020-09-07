import os
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.policies import policy_saver


def create_savers(metadata_path, agent, replay_buffer):
    global_step = tf.compat.v1.train.get_or_create_global_step()

    checkpoint_path = os.path.join(metadata_path, "saved", "checkpoint")
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_path,
        max_to_keep=3,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )

    saver = policy_saver.PolicySaver(agent.policy)

    return train_checkpointer, saver
