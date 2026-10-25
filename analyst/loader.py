"""CSV upload, validation and registration into DuckDB."""
from __future__ import annotations
import io, logging, os, re
from dataclasses import dataclass, field
import duckdb
import pandas as pd

log = logging.getLogger(__name__)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))


class DataValidationError(Exception):
    pass


@dataclass
class DataStore:
    con: duckdb.DuckDBPyConnection = field(default_factory=lambda: duckdb.connect(":memory:"))
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def add(self, filename: str, raw: bytes) -> str:
        df = parse_csv(filename, raw)
        name = unique_name(table_name(filename), self.frames)
        self.con.register("_tmp", df)
        self.con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _tmp')
        self.con.unregister("_tmp")
        self.frames[name] = df
        log.info("Loaded %s as table %s (%d rows)", filename, name, len(df))
        return name

    def remove(self, name: str) -> None:
        self.con.execute(f'DROP TABLE IF EXISTS "{name}"')
        self.frames.pop(name, None)

    def schema_text(self, sample_rows: int = 3) -> str:
        parts = []
        for name, df in self.frames.items():
            cols = ", ".join(f"{r[0]} ({r[1]})" for r in self.con.execute(f'DESCRIBE "{name}"').fetchall())
            parts.append(f'Table "{name}" ({len(df)} rows)\n  Columns: {cols}\n'
                         f'  Sample rows (CSV):\n{df.head(sample_rows).to_csv(index=False)}')
        return "\n".join(parts) or "(no tables loaded)"


def table_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    base = re.sub(r"\W+", "_", base).strip("_") or "data"
    return f"t_{base}" if base[0].isdigit() else base


def unique_name(name: str, existing) -> str:
    n, i = name, 2
    while n in existing:
        n, i = f"{name}_{i}", i + 1
    return n


def parse_csv(filename: str, raw: bytes) -> pd.DataFrame:
    if not filename.lower().endswith(".csv"):
        raise DataValidationError(f"{filename}: only .csv files are supported.")
    if not raw or not raw.strip():
        raise DataValidationError(f"{filename}: file is empty.")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise DataValidationError(f"{filename}: exceeds {MAX_MB} MB limit.")
    df, err = None, None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            break
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            err = e
    if df is None:
        raise DataValidationError(f"{filename}: could not parse CSV ({err}).")
    if df.empty or df.shape[1] == 0:
        raise DataValidationError(f"{filename}: no data rows found.")
    df.columns = _clean_columns(df.columns)
    for c in df.columns:  # parse date-like columns
        if pd.api.types.is_string_dtype(df[c]) and re.search(r"date|time|month|day", c):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[c] = parsed
    return df


def _clean_columns(cols) -> list[str]:
    out, seen = [], {}
    for c in cols:
        c = re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") or "col"
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    return out
