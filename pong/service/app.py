import click
from flask import Flask
from pong.service.blueprints.random_replay_buffer import create_random_replay_buffer


def create_app(replay_buffer_path):
    app = Flask(__name__)

    replay_buffer_blueprint = create_random_replay_buffer(replay_buffer_path)
    app.register_blueprint(replay_buffer_blueprint, url_prefix="/random/buffer")

    return app
