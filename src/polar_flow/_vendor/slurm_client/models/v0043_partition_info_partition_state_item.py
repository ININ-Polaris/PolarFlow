from enum import Enum


class V0043PartitionInfoPartitionStateItem(str, Enum):
    DOWN = "DOWN"
    DRAIN = "DRAIN"
    INACTIVE = "INACTIVE"
    UNKNOWN = "UNKNOWN"
    UP = "UP"

    def __str__(self) -> str:
        return str(self.value)
