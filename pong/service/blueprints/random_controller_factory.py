import random

from flask import request, Blueprint

from pong.constants import DIRECTIONS_TO_CHOOSE


def create_random_controller():
    blueprint = Blueprint("random", __name__)

    @blueprint.route("/act", methods=["POST"])
    def create_line():
        print(request.json)
        direction = random.choice(DIRECTIONS_TO_CHOOSE)
        return dict(action=direction), 200

    @blueprint.route("/finish", methods=["POST"])
    def finish():
        print(request.json)
        return dict(), 200

    return blueprint
