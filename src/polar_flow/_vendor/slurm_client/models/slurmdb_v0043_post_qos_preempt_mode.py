from enum import Enum


class SlurmdbV0043PostQosPreemptMode(str, Enum):
    CANCEL = "CANCEL"
    DISABLED = "DISABLED"
    GANG = "GANG"
    REQUEUE = "REQUEUE"
    SUSPEND = "SUSPEND"

    def __str__(self) -> str:
        return str(self.value)
