import random
from flask import request

from pong.replay_buffer import vectorize_state
from pong.service.blueprints.base_blueprint import BaseBlueprint

DIRECTIONS_TO_CHOOSE = ["Up", "Down", None]


def create_random_replay_buffer(replay_buffer_path):
    blueprint = BaseBlueprint(replay_buffer_path, "random_replay_buffer", __name__)

    @blueprint.route("/", methods=["POST"])
    def create_line():
        line = vectorize_state(request.json)
        blueprint.append(line)
        direction = random.choice(DIRECTIONS_TO_CHOOSE)
        return dict(action=direction), 200

    @blueprint.route("/finish", methods=["POST"])
    def finish():
        line = vectorize_state(request.json)
        blueprint.append(line)
        blueprint.flush()
        return dict(), 200

    return blueprint
