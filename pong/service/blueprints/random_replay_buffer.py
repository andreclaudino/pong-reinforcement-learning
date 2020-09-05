import random
from flask import Blueprint
from flask import request

from pong.replay_buffer import persist

DIRECTIONS_TO_CHOOSE = ["Up", "Down", None]


class RandomBlueprint(Blueprint):

    def __init__(self, replay_buffer_path, name, import_name):
        super().__init__(name, import_name)

        self.replay_buffer_path = replay_buffer_path
        self.episode_buffer = []

    def append(self, line):
        self.episode_buffer.append(line)

    def flush(self):
        persist(self.episode_buffer, self.replay_buffer_path)
        self.episode_buffer.clear()


def create_random_replay_buffer(replay_buffer_path):
    blueprint = RandomBlueprint(replay_buffer_path, "random_replay_buffer", __name__)

    @blueprint.route("/", methods=["POST"])
    def create_line():
        line = request.json
        blueprint.append(line)
        direction = random.choice(DIRECTIONS_TO_CHOOSE)
        return dict(action=direction), 200

    @blueprint.route("/finish", methods=["POST"])
    def finish():
        line = request.json
        blueprint.append(line)
        blueprint.flush()
        return dict(), 200

    return blueprint
