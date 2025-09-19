from enum import Enum


class V0043PartitionInfoMaximumsOversubscribeFlagsItem(str, Enum):
    FORCE = "force"

    def __str__(self) -> str:
        return str(self.value)
