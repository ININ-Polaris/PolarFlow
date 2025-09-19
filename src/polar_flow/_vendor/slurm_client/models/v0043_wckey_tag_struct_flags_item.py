from enum import Enum


class V0043WckeyTagStructFlagsItem(str, Enum):
    ASSIGNED_DEFAULT = "ASSIGNED_DEFAULT"

    def __str__(self) -> str:
        return str(self.value)
