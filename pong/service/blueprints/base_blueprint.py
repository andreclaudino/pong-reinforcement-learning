from flask import Blueprint
from pong.replay_buffer import persist


class BaseBlueprint(Blueprint):

    def __init__(self, replay_buffer_path, name, import_name):
        super().__init__(name, import_name)

        self.replay_buffer_path = replay_buffer_path
        self.episode_buffer = []

    def append(self, line):
        self.episode_buffer.append(line)

    def flush(self):
        persist(self.episode_buffer, self.replay_buffer_path)
        self.episode_buffer.clear()
