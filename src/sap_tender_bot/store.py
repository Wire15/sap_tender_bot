from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            since_iso TEXT,
            bootstrap INTEGER NOT NULL DEFAULT 0,
            dry_run INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ok',
            new_count INTEGER DEFAULT 0,
            updated_count INTEGER DEFAULT 0,
            flagged_count INTEGER DEFAULT 0,
            rejected_count INTEGER DEFAULT 0,
            watchlist_count INTEGER DEFAULT 0,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS tenders (
            tender_key TEXT PRIMARY KEY,
            source TEXT,
            uid TEXT,
            ref_id TEXT,
            solicitation_number TEXT,
            title TEXT,
            org TEXT,
            url TEXT,
            first_seen TEXT,
            last_seen TEXT,
            last_amendment_seen TEXT,
            last_close_date_seen TEXT,
            last_attachment_hash TEXT,
            last_payload TEXT
        );

        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            tender_key TEXT NOT NULL,
            uid TEXT,
            decision TEXT NOT NULL,
            score INTEGER,
            reject_reason TEXT,
            events TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id),
            FOREIGN KEY(tender_key) REFERENCES tenders(tender_key)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tender_key TEXT NOT NULL,
            uid TEXT,
            created_at TEXT NOT NULL,
            status TEXT,
            notes TEXT,
            tags TEXT,
            FOREIGN KEY(tender_key) REFERENCES tenders(tender_key)
        );

        CREATE INDEX IF NOT EXISTS idx_runs_finished_at ON runs(finished_at);
        CREATE INDEX IF NOT EXISTS idx_tenders_last_seen ON tenders(last_seen);
        CREATE INDEX IF NOT EXISTS idx_decisions_run_id ON decisions(run_id);
        CREATE INDEX IF NOT EXISTS idx_decisions_tender_key ON decisions(tender_key);
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS attachment_cache (
            fingerprint TEXT PRIMARY KEY,
            url_norm TEXT NOT NULL,
            url TEXT,
            etag TEXT,
            last_modified TEXT,
            content_hash TEXT,
            mime_type TEXT,
            size_bytes INTEGER,
            fetched_at TEXT,
            text_path TEXT,
            text_chars INTEGER,
            summary_json TEXT,
            summary_cached_at TEXT,
            requirements_json TEXT,
            summary_model TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_attachment_cache_url_meta
            ON attachment_cache(url_norm, etag, last_modified);
        CREATE INDEX IF NOT EXISTS idx_attachment_cache_url_hash
            ON attachment_cache(url_norm, content_hash);
        """,
    ),
    (
        3,
        """
        ALTER TABLE attachment_cache ADD COLUMN structured_summary_json TEXT;
        ALTER TABLE attachment_cache ADD COLUMN source_sections_json TEXT;
        ALTER TABLE attachment_cache ADD COLUMN selected_chunks_json TEXT;
        """,
    ),
]


def _apply_migrations(conn: sqlite3.Connection, logger: logging.Logger) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {
        row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for version, sql in MIGRATIONS:
        if version in applied:
            continue
        logger.info("Applying DB migration %s", version)
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, _now_utc_iso()),
        )
    conn.commit()


@dataclass
class RunRecord:
    started_at: str
    since_iso: Optional[str]
    bootstrap: bool
    dry_run: bool


class TenderStore:
    def __init__(self, db_path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.db_path = Path(db_path)
        self.logger = logger or logging.getLogger(__name__)
        self.conn: Optional[sqlite3.Connection] = None

    def open(self) -> "TenderStore":
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        _apply_migrations(self.conn, self.logger)
        return self

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_last_run_iso(self) -> Optional[str]:
        if not self.conn:
            raise RuntimeError("Store is not open")
        row = self.conn.execute(
            """
            SELECT finished_at
            FROM runs
            WHERE status = 'ok' AND dry_run = 0 AND finished_at IS NOT NULL
            ORDER BY finished_at DESC
            LIMIT 1
            """
        ).fetchone()
        return str(row["finished_at"]) if row else None

    def start_run(self, record: RunRecord) -> int:
        if not self.conn:
            raise RuntimeError("Store is not open")
        cur = self.conn.execute(
            """
            INSERT INTO runs (started_at, since_iso, bootstrap, dry_run)
            VALUES (?, ?, ?, ?)
            """,
            (
                record.started_at,
                record.since_iso,
                1 if record.bootstrap else 0,
                1 if record.dry_run else 0,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(
        self,
        run_id: int,
        *,
        finished_at: str,
        status: str = "ok",
        error: Optional[str] = None,
        new_count: int = 0,
        updated_count: int = 0,
        flagged_count: int = 0,
        rejected_count: int = 0,
        watchlist_count: int = 0,
    ) -> None:
        if not self.conn:
            raise RuntimeError("Store is not open")
        self.conn.execute(
            """
            UPDATE runs
            SET finished_at = ?,
                status = ?,
                error = ?,
                new_count = ?,
                updated_count = ?,
                flagged_count = ?,
                rejected_count = ?,
                watchlist_count = ?
            WHERE id = ?
            """,
            (
                finished_at,
                status,
                error,
                new_count,
                updated_count,
                flagged_count,
                rejected_count,
                watchlist_count,
                run_id,
            ),
        )
        self.conn.commit()

    def fetch_tender(self, tender_key: str) -> Optional[dict]:
        if not self.conn:
            raise RuntimeError("Store is not open")
        row = self.conn.execute(
            "SELECT * FROM tenders WHERE tender_key = ?",
            (tender_key,),
        ).fetchone()
        return dict(row) if row else None

    def upsert_tender(
        self,
        *,
        tender_key: str,
        source: str,
        uid: str,
        ref_id: str,
        solicitation_number: str,
        title: str,
        org: str,
        url: str,
        first_seen: str,
        last_seen: str,
        last_amendment_seen: Optional[str],
        last_close_date_seen: Optional[str],
        last_attachment_hash: Optional[str],
        payload: dict,
    ) -> None:
        if not self.conn:
            raise RuntimeError("Store is not open")
        payload_json = json.dumps(payload, ensure_ascii=False)
        self.conn.execute(
            """
            INSERT INTO tenders (
                tender_key, source, uid, ref_id, solicitation_number,
                title, org, url,
                first_seen, last_seen,
                last_amendment_seen, last_close_date_seen, last_attachment_hash,
                last_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tender_key) DO UPDATE SET
                source = excluded.source,
                uid = excluded.uid,
                ref_id = excluded.ref_id,
                solicitation_number = excluded.solicitation_number,
                title = excluded.title,
                org = excluded.org,
                url = excluded.url,
                first_seen = COALESCE(tenders.first_seen, excluded.first_seen),
                last_seen = excluded.last_seen,
                last_amendment_seen = COALESCE(excluded.last_amendment_seen, tenders.last_amendment_seen),
                last_close_date_seen = COALESCE(excluded.last_close_date_seen, tenders.last_close_date_seen),
                last_attachment_hash = COALESCE(excluded.last_attachment_hash, tenders.last_attachment_hash),
                last_payload = excluded.last_payload
            """,
            (
                tender_key,
                source,
                uid,
                ref_id,
                solicitation_number,
                title,
                org,
                url,
                first_seen,
                last_seen,
                last_amendment_seen,
                last_close_date_seen,
                last_attachment_hash,
                payload_json,
            ),
        )
        self.conn.commit()

    def record_decisions(self, rows: Iterable[dict]) -> None:
        if not self.conn:
            raise RuntimeError("Store is not open")
        rows = list(rows)
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT INTO decisions (
                run_id, tender_key, uid, decision, score, reject_reason, events, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.get("run_id"),
                    row.get("tender_key"),
                    row.get("uid"),
                    row.get("decision"),
                    row.get("score"),
                    row.get("reject_reason"),
                    row.get("events"),
                    row.get("created_at"),
                )
                for row in rows
            ],
        )
        self.conn.commit()

    def update_tender_payload(self, *, tender_key: str, payload: dict) -> None:
        if not self.conn:
            raise RuntimeError("Store is not open")
        payload_json = json.dumps(payload, ensure_ascii=False)
        self.conn.execute(
            "UPDATE tenders SET last_payload = ? WHERE tender_key = ?",
            (payload_json, tender_key),
        )
        self.conn.commit()

    def fetch_tender_payloads_between(
        self, *, start_iso: str, end_iso: str
    ) -> list[dict]:
        if not self.conn:
            raise RuntimeError("Store is not open")
        rows = self.conn.execute(
            """
            SELECT last_payload
            FROM tenders
            WHERE last_seen >= ? AND last_seen <= ?
            """,
            (start_iso, end_iso),
        ).fetchall()
        payloads: list[dict] = []
        for row in rows:
            raw = row["last_payload"] if row else None
            if not raw:
                continue
            try:
                payloads.append(json.loads(raw))
            except Exception:
                continue
        return payloads

    def fetch_recent_run_durations(self, limit: int = 10) -> list[int]:
        if not self.conn:
            raise RuntimeError("Store is not open")
        rows = self.conn.execute(
            """
            SELECT started_at, finished_at
            FROM runs
            WHERE status = 'ok' AND finished_at IS NOT NULL
            ORDER BY finished_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        durations: list[int] = []
        for row in rows:
            started = row["started_at"]
            finished = row["finished_at"]
            if not started or not finished:
                continue
            try:
                start_dt = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
                durations.append(int((end_dt - start_dt).total_seconds()))
            except Exception:
                continue
        return durations

    def fetch_attachment_cache_by_meta(
        self,
        *,
        url_norm: str,
        etag: Optional[str],
        last_modified: Optional[str],
    ) -> Optional[dict]:
        if not self.conn:
            raise RuntimeError("Store is not open")
        row = self.conn.execute(
            """
            SELECT * FROM attachment_cache
            WHERE url_norm = ? AND etag IS ? AND last_modified IS ?
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            (url_norm, etag, last_modified),
        ).fetchone()
        return dict(row) if row else None

    def fetch_attachment_cache_by_hash(
        self,
        *,
        url_norm: str,
        content_hash: str,
    ) -> Optional[dict]:
        if not self.conn:
            raise RuntimeError("Store is not open")
        row = self.conn.execute(
            """
            SELECT * FROM attachment_cache
            WHERE url_norm = ? AND content_hash = ?
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            (url_norm, content_hash),
        ).fetchone()
        return dict(row) if row else None

    def fetch_latest_attachment_meta(self, *, url_norm: str) -> Optional[dict]:
        if not self.conn:
            raise RuntimeError("Store is not open")
        row = self.conn.execute(
            """
            SELECT * FROM attachment_cache
            WHERE url_norm = ?
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            (url_norm,),
        ).fetchone()
        return dict(row) if row else None

    def upsert_attachment_cache(self, *, entry: dict) -> None:
        if not self.conn:
            raise RuntimeError("Store is not open")
        payload = {
            "fingerprint": entry.get("fingerprint"),
            "url_norm": entry.get("url_norm"),
            "url": entry.get("url"),
            "etag": entry.get("etag"),
            "last_modified": entry.get("last_modified"),
            "content_hash": entry.get("content_hash"),
            "mime_type": entry.get("mime_type"),
            "size_bytes": entry.get("size_bytes"),
            "fetched_at": entry.get("fetched_at"),
            "text_path": entry.get("text_path"),
            "text_chars": entry.get("text_chars"),
            "summary_json": entry.get("summary_json"),
            "summary_cached_at": entry.get("summary_cached_at"),
            "requirements_json": entry.get("requirements_json"),
            "summary_model": entry.get("summary_model"),
            "structured_summary_json": entry.get("structured_summary_json"),
            "source_sections_json": entry.get("source_sections_json"),
            "selected_chunks_json": entry.get("selected_chunks_json"),
        }
        self.conn.execute(
            """
            INSERT INTO attachment_cache (
                fingerprint, url_norm, url, etag, last_modified, content_hash,
                mime_type, size_bytes, fetched_at, text_path, text_chars,
                summary_json, summary_cached_at, requirements_json, summary_model,
                structured_summary_json, source_sections_json, selected_chunks_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET
                url_norm = excluded.url_norm,
                url = excluded.url,
                etag = excluded.etag,
                last_modified = excluded.last_modified,
                content_hash = excluded.content_hash,
                mime_type = excluded.mime_type,
                size_bytes = excluded.size_bytes,
                fetched_at = excluded.fetched_at,
                text_path = COALESCE(excluded.text_path, attachment_cache.text_path),
                text_chars = COALESCE(excluded.text_chars, attachment_cache.text_chars),
                summary_json = COALESCE(excluded.summary_json, attachment_cache.summary_json),
                summary_cached_at = COALESCE(
                    excluded.summary_cached_at,
                    attachment_cache.summary_cached_at
                ),
                requirements_json = COALESCE(
                    excluded.requirements_json,
                    attachment_cache.requirements_json
                ),
                summary_model = COALESCE(excluded.summary_model, attachment_cache.summary_model),
                structured_summary_json = COALESCE(
                    excluded.structured_summary_json,
                    attachment_cache.structured_summary_json
                ),
                source_sections_json = COALESCE(
                    excluded.source_sections_json,
                    attachment_cache.source_sections_json
                ),
                selected_chunks_json = COALESCE(
                    excluded.selected_chunks_json,
                    attachment_cache.selected_chunks_json
                )
            """,
            (
                payload["fingerprint"],
                payload["url_norm"],
                payload["url"],
                payload["etag"],
                payload["last_modified"],
                payload["content_hash"],
                payload["mime_type"],
                payload["size_bytes"],
                payload["fetched_at"],
                payload["text_path"],
                payload["text_chars"],
                payload["summary_json"],
                payload["summary_cached_at"],
                payload["requirements_json"],
                payload["summary_model"],
                payload["structured_summary_json"],
                payload["source_sections_json"],
                payload["selected_chunks_json"],
            ),
        )
        self.conn.commit()
