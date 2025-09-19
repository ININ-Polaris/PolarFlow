from enum import Enum


class V0043JobDescMsgOpenModeItem(str, Enum):
    APPEND = "APPEND"
    TRUNCATE = "TRUNCATE"

    def __str__(self) -> str:
        return str(self.value)
