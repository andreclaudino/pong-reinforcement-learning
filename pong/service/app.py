from multiprocessing import Queue

from flask import Flask

from pong.service.blueprints import create_mailboxed_controller
from pong.service.blueprints import create_random_controller


def create_app(agent_mailbox: Queue, service_mailbox: Queue):
    app = Flask(__name__)

    random_environment_controller = create_random_controller()
    app.register_blueprint(random_environment_controller, url_prefix="/random/play")

    model_environment_controller = create_mailboxed_controller(agent_mailbox, service_mailbox)
    app.register_blueprint(model_environment_controller, url_prefix="/agent/play")

    return app
