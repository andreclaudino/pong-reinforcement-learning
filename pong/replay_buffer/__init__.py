import uuid
import os
import numpy as np


MAX_CHUNK_SIZE = 1000


def persist(buffer, base_path, data_dir=None):
    data_dir = data_dir if data_dir else str(uuid.uuid4())
    data_path = os.path.join(base_path, data_dir)
    os.makedirs(data_path, exist_ok=True)

    for chunk in _chunkfy(buffer, MAX_CHUNK_SIZE):
        file_name = os.path.join(data_path, str(uuid.uuid4()))
        np.savez(file_name, chunk)


def _chunkfy(date, size):
    for i in range(0, len(date), size):
        yield date[i:i+size]
