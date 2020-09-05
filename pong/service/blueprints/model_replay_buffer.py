import random
from flask import Blueprint
from flask import request

from pong.ia.model.utils import choose_direction
from pong.replay_buffer import vectorize_state


def create_replay_buffer(model):
    blueprint = Blueprint("replay_buffer", __name__)

    @blueprint.route("/", methods=["POST"])
    def create_line():
        line = vectorize_state(request.json)

        actions = model(line)
        direction = choose_direction(actions)
        return dict(action=direction), 200

    @blueprint.route("/finish", methods=["POST"])
    def finish():
        line = vectorize_state(request.json)
        print(line)
        return dict(), 200

    return blueprint