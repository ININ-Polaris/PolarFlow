from enum import Enum


class V0043ProcessExitCodeVerboseStatusItem(str, Enum):
    CORE_DUMPED = "CORE_DUMPED"
    ERROR = "ERROR"
    INVALID = "INVALID"
    PENDING = "PENDING"
    SIGNALED = "SIGNALED"
    SUCCESS = "SUCCESS"

    def __str__(self) -> str:
        return str(self.value)
