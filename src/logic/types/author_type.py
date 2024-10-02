from __future__ import annotations


from dataclasses import dataclass


@dataclass(frozen=True)
class Author:
    name: str
    id: str
    count: int

    @staticmethod
    def from_json(data: dict) -> Author:
        return Author(name=data['name'], id=data['id'], count=data['count'])
