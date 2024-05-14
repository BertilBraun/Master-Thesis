import sqlite3

from enum import Enum
from typing import Dict, List


class EvaluationType(Enum):
    EXPERT = 'expert'
    AUTOMATIC = 'automatic'


class DPODatabase:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup()

    def setup(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                chosen TEXT,
                rejected TEXT,
                evaluation_type TEXT,
                external_id OPTIONAL TEXT,
                author_name TEXT
            )
        """
        )
        self.conn.commit()

    def add_entry(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        eval_type: EvaluationType,
        author_name: str,
        external_id: str | None = None,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO preferences (prompt, chosen, rejected, evaluation_type, external_id, author_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (prompt, chosen, rejected, eval_type.value, external_id, author_name),
        )
        self.conn.commit()

    def get_entries_by_type(self, eval_type: EvaluationType) -> Dict[str, List[str]]:
        self.cursor.execute(
            'SELECT prompt, chosen, rejected FROM preferences WHERE evaluation_type=?',
            (eval_type.value,),
        )
        rows = self.cursor.fetchall()

        data = {'prompt': [], 'chosen': [], 'rejected': []}
        for prompt, chosen, rejected in rows:
            data['prompt'].append(prompt)
            data['chosen'].append(chosen)
            data['rejected'].append(rejected)

        return data

    def check_existence_by_external_id(self, external_id: str) -> bool:
        self.cursor.execute(
            'SELECT 1 FROM preferences WHERE external_id=?',
            (external_id,),
        )
        exists = self.cursor.fetchone() is not None
        return exists

    def check_existence_by_author_name_and_eval_type(self, author_name: str, eval_type: EvaluationType) -> bool:
        self.cursor.execute(
            'SELECT 1 FROM preferences WHERE author_name=? AND evaluation_type=?',
            (author_name, eval_type.value),
        )
        exists = self.cursor.fetchone() is not None
        return exists
