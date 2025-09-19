from enum import Enum


class SlurmdbV0043GetClusterFlags(str, Enum):
    DELETED = "DELETED"
    EXTERNAL = "EXTERNAL"
    FEDERATION = "FEDERATION"
    MULTIPLE_SLURMD = "MULTIPLE_SLURMD"
    REGISTERING = "REGISTERING"

    def __str__(self) -> str:
        return str(self.value)
