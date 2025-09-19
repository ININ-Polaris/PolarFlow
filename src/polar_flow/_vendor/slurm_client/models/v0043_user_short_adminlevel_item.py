from enum import Enum


class V0043UserShortAdminlevelItem(str, Enum):
    ADMINISTRATOR = "Administrator"
    NONE = "None"
    NOT_SET = "Not Set"
    OPERATOR = "Operator"

    def __str__(self) -> str:
        return str(self.value)
