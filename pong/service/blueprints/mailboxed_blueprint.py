from multiprocessing import Queue
from typing import Sequence

from flask import Blueprint

from pong.constants import DIRECTIONS_TO_CHOOSE, START_MESSAGE
from pong.messages import Observable, Action

MAX_MESSAGE_ATTEMPS = 1000


class MailboxedBlueprint(Blueprint):

    def __init__(self, name: str, agent_mailbox: Queue, service_mailbox: Queue):
        super().__init__(name, __name__)

        self._inbox = service_mailbox
        self._agent_mailbox = agent_mailbox

    def ask(self, observable: Sequence[float], reward: float) -> str:
        message = Observable(observable, reward, finished=False)
        return self._infer(message)

    def finish(self, observable: Sequence[float], reward: float):
        message = Observable(observable, reward, finished=True)
        return self._infer(message)

    def _send(self, message: Observable):
        self._agent_mailbox.put(message)

    def _receive(self) -> Action:
        count = 0
        while self._inbox.empty():
            print("waiting...", end="\r")
            count += 1

            if count >= MAX_MESSAGE_ATTEMPS:
                return Action(0, None)

        return self._inbox.get()

    def _infer(self, observable: Observable) -> str:
        self._send(observable)
        action = self._receive()
        if action:
            return DIRECTIONS_TO_CHOOSE[action.value]
        else:
            return START_MESSAGE
