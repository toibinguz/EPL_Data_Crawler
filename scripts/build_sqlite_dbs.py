"""
scripts/build_sqlite_dbs.py
Dump all files from clean/ into clean.db and all files from raw_data/ into raw_data.db (SQLite).
- .csv and .csv.gz files -> individual SQLite tables (chunked import, with ragged-CSV fallback).
- .json files -> stored in `raw_json_files` table (queryable via SQLite json_extract).
- .tar.gz archives -> skipped (already unpacked into matches/ and cache_matches/).
- `_catalog` table -> lists every imported file, target table, row count, and any errors.
"""

import csv
import gzip
import re
import time
import sqlite3
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent


def sanitize_table_name(rel_path: Path, strip_ext: bool = True) -> str:
    s = str(rel_path).replace("\\", "/")
    if strip_ext:
        if s.endswith(".csv.gz"):
            s = s[:-7]
        elif s.endswith(".csv"):
            s = s[:-4]
    s = s.replace("/", "__")
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    if s and s[0].isdigit():
        s = "t_" + s
    return s


def deduplicate_columns(cols) -> list[str]:
    seen = {}
    out = []
    for c in cols:
        c_str = str(c).strip() or "col"
        key = c_str.lower()
        if key not in seen:
            seen[key] = 1
            out.append(c_str)
        else:
            seen[key] += 1
            out.append(f"{c_str}_{seen[key]}")
    return out


def read_ragged_csv(file_path: Path) -> pd.DataFrame:
    opener = gzip.open if file_path.name.endswith(".gz") else open
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            with opener(file_path, "rt", encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    max_len = max(len(r) for r in rows)
    header = list(rows[0])
    while len(header) < max_len:
        header.append(f"extra_col_{len(header) + 1}")
    data = [r + [""] * (max_len - len(r)) for r in rows[1:]]
    return pd.DataFrame(data, columns=header)


def read_csv_chunks(file_path: Path, chunksize: int = 50000):
    compression = "gzip" if file_path.name.endswith(".gz") else None
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            reader = pd.read_csv(
                file_path,
                chunksize=chunksize,
                low_memory=False,
                encoding=enc,
                compression=compression,
            )
            first_chunk = next(reader, None)
            if first_chunk is None:
                return
            yield first_chunk
            for chunk in reader:
                yield chunk
            return
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except pd.errors.ParserError:
            df = read_ragged_csv(file_path)
            if not df.empty:
                yield df
            return
    if last_err:
        raise last_err


def init_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB cache
    conn.execute(
        """
        CREATE TABLE _catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rel_path TEXT,
            file_type TEXT,
            table_name TEXT,
            row_count INTEGER,
            size_bytes INTEGER,
            status TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE raw_json_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            rel_path TEXT UNIQUE,
            file_name TEXT,
            size_bytes INTEGER,
            content TEXT
        )
        """
    )
    return conn


def build_db_from_dir(source_dir: Path, db_path: Path):
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"Building {db_path.name} from {source_dir.name}/ ...")
    print(f"{'='*70}")

    conn = init_db(db_path)
    all_files = sorted(p for p in source_dir.rglob("*") if p.is_file())

    csv_count = 0
    json_batch = []
    json_total = 0
    skipped = 0
    errors = 0

    for fpath in all_files:
        rel = fpath.relative_to(source_dir)
        rel_str = str(rel).replace("\\", "/")
        size_bytes = fpath.stat().st_size
        name_lower = fpath.name.lower()

        if name_lower.endswith(".tar.gz"):
            skipped += 1
            conn.execute(
                "INSERT INTO _catalog (rel_path, file_type, table_name, row_count, size_bytes, status) VALUES (?, ?, ?, ?, ?, ?)",
                (rel_str, "tar.gz", None, 0, size_bytes, "skipped_archive"),
            )
            continue

        if name_lower.endswith(".csv") or name_lower.endswith(".csv.gz"):
            table_name = sanitize_table_name(rel)
            total_rows = 0
            try:
                first = True
                clean_cols = None
                for chunk in read_csv_chunks(fpath):
                    if first:
                        clean_cols = deduplicate_columns(chunk.columns)
                        first = False
                    chunk.columns = clean_cols
                    chunk.to_sql(
                        table_name,
                        conn,
                        if_exists="append" if total_rows > 0 else "replace",
                        index=False,
                    )
                    total_rows += len(chunk)
                status = "ok" if total_rows > 0 else "empty_csv"
                csv_count += 1
                print(f"  [CSV {csv_count}] {rel_str} -> {table_name} ({total_rows:,} rows)")
            except Exception as e:
                errors += 1
                status = f"error: {e}"
                print(f"  [ERR] {rel_str}: {e}")

            conn.execute(
                "INSERT INTO _catalog (rel_path, file_type, table_name, row_count, size_bytes, status) VALUES (?, ?, ?, ?, ?, ?)",
                (rel_str, "csv", table_name, total_rows, size_bytes, status),
            )

        elif name_lower.endswith(".json"):
            source = rel.parts[0] if len(rel.parts) > 1 else source_dir.name
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                json_batch.append((source, rel_str, fpath.name, size_bytes, content))
                json_total += 1
                if len(json_batch) >= 500:
                    conn.executemany(
                        "INSERT INTO raw_json_files (source, rel_path, file_name, size_bytes, content) VALUES (?, ?, ?, ?, ?)",
                        json_batch,
                    )
                    json_batch.clear()
                    if json_total % 2000 == 0:
                        print(f"  [JSON] Inserted {json_total:,} JSON files...")
            except Exception as e:
                errors += 1
                print(f"  [ERR JSON] {rel_str}: {e}")
                conn.execute(
                    "INSERT INTO _catalog (rel_path, file_type, table_name, row_count, size_bytes, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (rel_str, "json", "raw_json_files", 0, size_bytes, f"error: {e}"),
                )
        else:
            skipped += 1

    if json_batch:
        conn.executemany(
            "INSERT INTO raw_json_files (source, rel_path, file_name, size_bytes, content) VALUES (?, ?, ?, ?, ?)",
            json_batch,
        )
        json_batch.clear()

    if json_total > 0:
        conn.execute(
            "INSERT INTO _catalog (rel_path, file_type, table_name, row_count, size_bytes, status) VALUES (?, ?, ?, ?, ?, ?)",
            (f"{source_dir.name}/**/*.json", "json_collection", "raw_json_files", json_total, 0, "ok"),
        )
        conn.execute("CREATE INDEX idx_json_source_file ON raw_json_files(source, file_name);")

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    db_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"\nDone: {db_path.name} ({db_mb:.2f} MB) in {elapsed:.1f}s")
    print(f"  CSVs imported: {csv_count}, JSONs imported: {json_total:,}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    build_db_from_dir(ROOT_DIR / "clean", ROOT_DIR / "clean.db")
    build_db_from_dir(ROOT_DIR / "raw_data", ROOT_DIR / "raw_data.db")
