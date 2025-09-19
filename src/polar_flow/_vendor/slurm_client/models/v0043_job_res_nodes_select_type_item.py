from enum import Enum


class V0043JobResNodesSelectTypeItem(str, Enum):
    AVAILABLE = "AVAILABLE"
    ONE_ROW = "ONE_ROW"
    RESERVED = "RESERVED"

    def __str__(self) -> str:
        return str(self.value)
