from enum import Enum


class V0043JobResSelectTypeItem(str, Enum):
    BOARD = "BOARD"
    CORE = "CORE"
    CORE_DEFAULT_DIST_BLOCK = "CORE_DEFAULT_DIST_BLOCK"
    CPU = "CPU"
    LINEAR = "LINEAR"
    LLN = "LLN"
    MEMORY = "MEMORY"
    ONE_TASK_PER_CORE = "ONE_TASK_PER_CORE"
    PACK_NODES = "PACK_NODES"
    SOCKET = "SOCKET"

    def __str__(self) -> str:
        return str(self.value)
