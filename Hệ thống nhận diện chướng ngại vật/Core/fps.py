import time
from collections import deque

class FPSCounter:
    def __init__(self, buffer_size=20):
        self.buffer = deque(maxlen=buffer_size)
        self.prev_time = time.time()

    def update(self):
        now = time.time()
        fps = 1 / (now - self.prev_time)
        self.prev_time = now

        self.buffer.append(fps)
        return sum(self.buffer) / len(self.buffer)
