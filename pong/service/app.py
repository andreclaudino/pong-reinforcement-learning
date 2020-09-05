from flask import Flask
from pong.service.blueprints.replay_buffer import create_replay_buffer


def create_app():
    app = Flask(__name__)

    replay_buffer_blueprint = create_replay_buffer()
    app.register_blueprint(replay_buffer_blueprint, url_prefix="/buffer")

    return app
