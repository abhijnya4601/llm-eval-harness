import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from dataclasses import asdict

from .providers.base import GenerationResult

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).parent.parent / "cache" / "cache.db"


class DiskCache:

    def __init__(self, db_path: Path = _DEFAULT_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def make_key(
        provider: str,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        raw = f"{provider}:{model}:{prompt}:{max_tokens}:{temperature}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> GenerationResult | None:
        row = self._conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            logger.debug("Cache miss: %s", key[:16])
            return None
        logger.debug("Cache hit: %s", key[:16])
        data = json.loads(row[0])
        return GenerationResult(**data)

    def set(self, key: str, result: GenerationResult) -> None:
        value = json.dumps(asdict(result))
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
