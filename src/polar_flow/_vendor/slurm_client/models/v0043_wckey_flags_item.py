from enum import Enum


class V0043WckeyFlagsItem(str, Enum):
    DELETED = "DELETED"

    def __str__(self) -> str:
        return str(self.value)
