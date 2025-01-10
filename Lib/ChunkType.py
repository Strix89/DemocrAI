from enum import Enum

class ChunkType(Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    #sAGENTIC = "agentic"

    @classmethod
    def from_string(cls, value: str):
        try:
            return next(member for member in cls if member.value.lower() == value.lower())
        except StopIteration:
            return cls.RECURSIVE