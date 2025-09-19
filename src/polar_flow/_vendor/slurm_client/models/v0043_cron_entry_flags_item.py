from enum import Enum


class V0043CronEntryFlagsItem(str, Enum):
    WILD_DAY_OF_MONTH = "WILD_DAY_OF_MONTH"
    WILD_DAY_OF_WEEK = "WILD_DAY_OF_WEEK"
    WILD_HOUR = "WILD_HOUR"
    WILD_MINUTE = "WILD_MINUTE"
    WILD_MONTH = "WILD_MONTH"

    def __str__(self) -> str:
        return str(self.value)
