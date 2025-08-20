from enum import Enum


class Actions(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1
    def __str__(self):
        return self.name  # just "BUY", "SELL", "HOLD"