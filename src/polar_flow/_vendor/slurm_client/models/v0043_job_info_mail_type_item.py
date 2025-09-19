from enum import Enum


class V0043JobInfoMailTypeItem(str, Enum):
    ARRAY_TASKS = "ARRAY_TASKS"
    BEGIN = "BEGIN"
    END = "END"
    FAIL = "FAIL"
    INVALID_DEPENDENCY = "INVALID_DEPENDENCY"
    REQUEUE = "REQUEUE"
    STAGE_OUT = "STAGE_OUT"
    TIME100 = "TIME=100%"
    TIME50 = "TIME=50%"
    TIME80 = "TIME=80%"
    TIME90 = "TIME=90%"

    def __str__(self) -> str:
        return str(self.value)
