from enum import Enum


class V0043JobDescMsgX11Item(str, Enum):
    BATCH_NODE = "BATCH_NODE"
    FIRST_NODE = "FIRST_NODE"
    FORWARD_ALL_NODES = "FORWARD_ALL_NODES"
    LAST_NODE = "LAST_NODE"

    def __str__(self) -> str:
        return str(self.value)
