from typing import TypedDict


class Preference(TypedDict):
    prompt: str
    chosen: str
    rejected: str
