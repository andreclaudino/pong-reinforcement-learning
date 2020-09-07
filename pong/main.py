from multiprocessing import Process
from multiprocessing import Queue
from time import sleep

import click

from pong.ia.model.agent import create_pong_agent
from pong.ia.model.pong_environment import create_environment
from pong.ia.model.replay_buffer import create_replay_buffer
from pong.ia.train import train
from pong.service.app import create_app

REPLAY_BUFFER_BATCH_SIZE = 32


@click.command()
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default="8080")
@click.option("--dense-layers-sizes", type=str, default="64,64,64")
@click.option("--learning_rate", type=float, default=0.01)
@click.option("--max-buffer-lenght", type=int, default=500)
@click.option("--discount-factor", type=float, default=0.9)
@click.option("--num-episodes", type=int, default=10)
@click.option("--save-path", type=click.Path(file_okay=False, dir_okay=True, writable=True, allow_dash=True), default="output")
@click.option("--validate-step", type=int, default=10)
@click.option("--randomize-step", type=int, default=10)
def main(host: str, port: int, dense_layers_sizes: str, learning_rate: float, max_buffer_lenght,
         discount_factor: float, num_episodes: int, save_path: str, validate_step: int, randomize_step: int):
    dense_layers_sizes_list = [int(_) for _ in dense_layers_sizes.split(",")]

    service_mailbox_common = Queue()
    agent_mailbox_common = Queue()

    def training_process_function(agent_mailbox, service_mailbox):

        pong_environment = create_environment(agent_mailbox, service_mailbox, discount_factor)
        agent = create_pong_agent(pong_environment, dense_layers_sizes_list, learning_rate)

        replay_buffer = create_replay_buffer(agent, pong_environment, max_buffer_lenght)
        train(agent, pong_environment, replay_buffer, num_episodes, REPLAY_BUFFER_BATCH_SIZE, save_path,
              randomize_step, validate_step)
    training_process = Process(target=training_process_function, args=(agent_mailbox_common, service_mailbox_common))

    def server_process_function(agent_mailbox, service_mailbox):
        app = create_app(agent_mailbox, service_mailbox)
        app.run(host=host, port=port, threaded=False)
    server_process = Process(target=server_process_function, args=(agent_mailbox_common, service_mailbox_common))

    server_process.start()
    sleep(1)
    training_process.start()
    training_process.join()
    server_process.join()


if __name__ == '__main__':
    main()
