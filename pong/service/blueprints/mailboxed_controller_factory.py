from queue import Queue

from flask import request

from pong.service.blueprints.mailboxed_blueprint import MailboxedBlueprint

AGENT_BLUEPRINT_NAME = "agent"


def create_mailboxed_controller(service_mailbox: Queue, agent_mailbox: Queue):
    blueprint = MailboxedBlueprint(AGENT_BLUEPRINT_NAME, service_mailbox, agent_mailbox)

    @blueprint.route("/act", methods=["POST"])
    def create_line():
        observable = request.json["observation"]
        reward = request.json["score"]

        direction = blueprint.ask(observable, reward)
        return dict(action=direction), 200

    @blueprint.route("/finish", methods=["POST"])
    def finish():
        print(request.json)
        return dict(), 200

    return blueprint
