from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("idx", len(rows))
            if "problem" not in row:
                raise ValueError(f"{path}:{line_no}: missing problem")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
