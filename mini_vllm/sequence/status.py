from enum import Enum, auto


class Status(Enum):
    WAITING = auto()
    PREFILL = auto()
    RUNNING = auto()
    FINISHED = auto()
    REJECTED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()

    def __repr__(self) -> str:
        return self.name
