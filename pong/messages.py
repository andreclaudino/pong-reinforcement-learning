from typing import Sequence


class Observable:
    def __init__(self, observable: Sequence[float], reward: float, finished: bool):
        self._observable = observable
        self._reward = reward
        self._finished = finished

    @property
    def observable(self) -> Sequence[float]:
        return self._observable

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def finished(self) -> bool:
        return self._finished


class Action:

    def __init__(self, value: int):
        self._value = value

    @property
    def value(self):
        return self._value
