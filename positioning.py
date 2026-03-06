import numpy as np
import pandas as pd


class PositionBuilder:

    def __init__(self, direction="long", threshold=0.5):
        self.direction = direction
        self.threshold = threshold

    def build(self, signal):
        pos = 0 * signal

        if self.direction == "long":
            pos[signal > self.threshold] = 1

        elif self.direction == "short":
            pos[signal < (1 - self.threshold)] = -1

        elif self.direction == "both":
            pos[signal > self.threshold] = 1
            pos[signal < (1 - self.threshold)] = -1

        return pos

