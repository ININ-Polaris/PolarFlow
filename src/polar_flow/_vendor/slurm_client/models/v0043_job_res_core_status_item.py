from enum import Enum


class V0043JobResCoreStatusItem(str, Enum):
    ALLOCATED = "ALLOCATED"
    INVALID = "INVALID"
    IN_USE = "IN_USE"
    UNALLOCATED = "UNALLOCATED"

    def __str__(self) -> str:
        return str(self.value)
