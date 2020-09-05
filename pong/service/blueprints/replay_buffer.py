from time import sleep
import random
from flask import Blueprint
from flask import request


DIRECTIONS_TO_CHOOSE = ["Up", "Down", None]


def create_replay_buffer():
    blueprint = Blueprint("replay_buffer", __name__)

    @blueprint.route("/", methods=["POST"])
    def create_line():
        line = request.json
        print(line)

        direction = random.choice(DIRECTIONS_TO_CHOOSE)
        return dict(action=direction), 200

    @blueprint.route("/finish", methods=["POST"])
    def finish():
        line = request.json
        print(line)
        return dict(), 200

    return blueprint
