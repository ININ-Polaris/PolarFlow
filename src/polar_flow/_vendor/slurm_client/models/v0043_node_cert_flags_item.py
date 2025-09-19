from enum import Enum


class V0043NodeCertFlagsItem(str, Enum):
    TOKEN_SET = "TOKEN_SET"

    def __str__(self) -> str:
        return str(self.value)
