"""Structured training metrics (CSV + JSONL).

Enabled when YAML sets ``metrics_log: metrics`` (basename without extension)
or env ``STAF_METRICS_LOG`` is set. Writes ``<base>.csv`` and ``<base>.jsonl``.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


_FIELDNAMES = (
    "global_step",
    "epoch",
    "rmse_e",
    "rmse_f",
    "loss_tot",
    "lr_net",
    "lr_finger",
)


class MetricsLog:
    def __init__(self, basenames: Optional[str] = None):
        raw = basenames or os.environ.get("STAF_METRICS_LOG") or ""
        raw = str(raw).strip()
        self.enabled = bool(raw)
        self._csv = None
        self._jsonl = None
        self._writer = None
        if not self.enabled:
            return
        base = Path(raw)
        if base.suffix in (".csv", ".jsonl", ".json"):
            base = base.with_suffix("")
        csv_path = Path(str(base) + ".csv")
        jsonl_path = Path(str(base) + ".jsonl")
        new_csv = not csv_path.exists()
        self._csv = csv_path.open("a", newline="")
        self._jsonl = jsonl_path.open("a")
        self._writer = csv.DictWriter(self._csv, fieldnames=_FIELDNAMES)
        if new_csv:
            self._writer.writeheader()
            self._csv.flush()

    def log(self, **fields: Any) -> None:
        if not self.enabled or self._writer is None:
            return
        row: Dict[str, Any] = {k: fields.get(k) for k in _FIELDNAMES}
        self._writer.writerow(row)
        self._csv.flush()
        self._jsonl.write(json.dumps(row) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        if self._csv is not None:
            self._csv.close()
        if self._jsonl is not None:
            self._jsonl.close()
