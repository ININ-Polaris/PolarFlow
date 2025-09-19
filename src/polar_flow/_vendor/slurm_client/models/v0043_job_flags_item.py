from enum import Enum


class V0043JobFlagsItem(str, Enum):
    CLEAR_SCHEDULING = "CLEAR_SCHEDULING"
    NONE = "NONE"
    NOT_SET = "NOT_SET"
    STARTED_ON_BACKFILL = "STARTED_ON_BACKFILL"
    STARTED_ON_SCHEDULE = "STARTED_ON_SCHEDULE"
    STARTED_ON_SUBMIT = "STARTED_ON_SUBMIT"
    START_RECEIVED = "START_RECEIVED"

    def __str__(self) -> str:
        return str(self.value)
