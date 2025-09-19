from enum import Enum


class V0043JobDescMsgProfileItem(str, Enum):
    ENERGY = "ENERGY"
    LUSTRE = "LUSTRE"
    NETWORK = "NETWORK"
    NONE = "NONE"
    NOT_SET = "NOT_SET"
    TASK = "TASK"

    def __str__(self) -> str:
        return str(self.value)
