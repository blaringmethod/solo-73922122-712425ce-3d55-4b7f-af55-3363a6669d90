"""SQLite persistence for the harvester.

The published catalogue and the harvest watermark live in the same SQLite
database.  A harvest round writes into ``run_records`` (a per-run staging
area) and only the final commit transaction copies staging into ``catalog``
and advances the watermark.  This is what makes a failed or interrupted
round a no-op for readers of the published catalogue.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def utc_now_millis() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def split_creators(creator: str | None) -> list[str]:
    """Split a stored creator field into individual attribution strings.

    The harvester stores ``dc:creator`` values joined by ``"; "``; each
    non-empty segment is one attribution occurrence, addressed by its
    position (0-based) in this list.  Splitting never merges or reorders:
    two identical names stay two distinct occurrences.
    """
    if not creator:
        return []
    return [part.strip() for part in creator.split(";") if part.strip()]


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()
        self._recover_interrupted_runs()
        self._ensure_invariants()

    # ------------------------------------------------------------------ schema
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS published_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    watermark TEXT,
                    updated_at TEXT,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    last_run_id TEXT
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (
                        status IN ('running', 'succeeded', 'failed', 'cancelled')
                    ),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    window_from TEXT,
                    window_until TEXT,
                    pages_fetched INTEGER NOT NULL DEFAULT 0,
                    records_seen INTEGER NOT NULL DEFAULT 0,
                    token_restarts INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    complete_list_size INTEGER,
                    phase TEXT,
                    last_error TEXT,
                    cancel_reason TEXT,
                    watermark_after TEXT
                );

                CREATE TABLE IF NOT EXISTS run_records (
                    run_id TEXT NOT NULL,
                    identifier TEXT NOT NULL,
                    title TEXT,
                    creator TEXT,
                    datestamp TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    seq INTEGER NOT NULL,
                    PRIMARY KEY (run_id, identifier),
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS catalog (
                    identifier TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    creator TEXT,
                    source_datestamp TEXT NOT NULL,
                    harvested_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_catalog_title ON catalog(title);

                -- 署名归并：人物编号。人物不按姓名自动合并，只由人工归并产生。
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                -- 一处署名（某条已发布书目的某个序位）只归一人。
                -- 归属只指向已发布目录；成功采集提交时在同一事务里调和：
                -- 序位与文字都未变的归属保留，其余解除；撤回即全部解除。
                CREATE TABLE IF NOT EXISTS attributions (
                    identifier TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    name_text TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    PRIMARY KEY (identifier, position),
                    FOREIGN KEY (person_id) REFERENCES persons(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (identifier) REFERENCES catalog(identifier)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_attributions_person
                    ON attributions(person_id);

                INSERT OR IGNORE INTO published_state
                    (id, watermark, updated_at, success_count, last_run_id)
                VALUES (1, NULL, NULL, 0, NULL);
                """
            )

    def _recover_interrupted_runs(self) -> None:
        """A previous process died.  No in-memory task can own these runs."""
        with self._lock:
            now = utc_now_millis()
            rows = self._conn.execute(
                "SELECT id FROM runs WHERE status = 'running'"
            ).fetchall()
            for row in rows:
                self._conn.execute(
                    """
                    UPDATE runs
                       SET status = 'failed',
                           finished_at = ?,
                           phase = 'failed',
                           last_error = COALESCE(
                               last_error,
                               '采集进程中断，本轮未提交（已发布目录与水位保持不变）'
                           )
                     WHERE id = ? AND status = 'running'
                    """,
                    (now, row["id"]),
                )
                self._conn.execute(
                    "DELETE FROM run_records WHERE run_id = ?", (row["id"],)
                )
            self._conn.execute(
                "DELETE FROM meta WHERE key = 'active_run_id'"
            )

    def _ensure_invariants(self) -> None:
        """At most one round may be ``running``, even across processes.

        Created after recovery so a legacy database with stray ``running``
        rows cannot make startup fail.
        """
        with self._lock:
            self._conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_one_active
                    ON runs(status) WHERE status = 'running'
                """
            )

    # ----------------------------------------------------------------- helpers
    def _meta_get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def _meta_set(self, key: str, value: str | None) -> None:
        if value is None:
            self._conn.execute("DELETE FROM meta WHERE key = ?", (key,))
        else:
            self._conn.execute(
                """
                INSERT INTO meta(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    # -------------------------------------------------------------------- runs
    def create_run(self) -> dict[str, Any]:
        with self._lock:
            run_id = uuid.uuid4().hex
            now = utc_now_millis()
            state = self._conn.execute(
                "SELECT watermark FROM published_state WHERE id = 1"
            ).fetchone()
            watermark = state["watermark"] if state else None
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO runs
                        (id, status, started_at, window_from, phase)
                    VALUES (?, 'running', ?, ?, 'connecting')
                    """,
                    (run_id, now, watermark),
                )
                self._meta_set("active_run_id", run_id)
                self._conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._conn.execute("ROLLBACK")
                raise RuntimeError("已有采集轮次在运行") from exc
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            run = self.get_run(run_id)
            assert run is not None
            return run

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            run = dict(row)
            staged = self._conn.execute(
                "SELECT COUNT(*) AS n FROM run_records WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            run["staged_count"] = int(staged["n"])
            return run

    def get_active_run(self) -> dict[str, Any] | None:
        with self._lock:
            run_id = self._meta_get("active_run_id")
            if run_id:
                run = self.get_run(run_id)
                if run and run["status"] == "running":
                    return run
            row = self._conn.execute(
                """
                SELECT * FROM runs WHERE status = 'running'
                 ORDER BY started_at DESC LIMIT 1
                """
            ).fetchone()
            return self.get_run(row["id"]) if row else None

    def list_runs(self, limit: int = 15) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def is_run_running(self, run_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM runs WHERE id = ? AND status = 'running'",
                (run_id,),
            ).fetchone()
            return row is not None

    def set_window(self, run_id: str, until: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE runs SET window_until = ?, phase = 'fetching'
                 WHERE id = ? AND status = 'running'
                """,
                (until, run_id),
            )
            return cur.rowcount > 0

    def update_progress(
        self,
        run_id: str,
        *,
        pages_fetched: int,
        records_seen: int,
        token_restarts: int,
        attempt: int,
        complete_list_size: int | None,
        phase: str | None = None,
    ) -> bool:
        with self._lock:
            if phase is None:
                cur = self._conn.execute(
                    """
                    UPDATE runs
                       SET pages_fetched = ?,
                           records_seen = ?,
                           token_restarts = ?,
                           attempt = ?,
                           complete_list_size = ?
                     WHERE id = ? AND status = 'running'
                    """,
                    (
                        pages_fetched,
                        records_seen,
                        token_restarts,
                        attempt,
                        complete_list_size,
                        run_id,
                    ),
                )
            else:
                cur = self._conn.execute(
                    """
                    UPDATE runs
                       SET pages_fetched = ?,
                           records_seen = ?,
                           token_restarts = ?,
                           attempt = ?,
                           complete_list_size = ?,
                           phase = ?
                     WHERE id = ? AND status = 'running'
                    """,
                    (
                        pages_fetched,
                        records_seen,
                        token_restarts,
                        attempt,
                        complete_list_size,
                        phase,
                        run_id,
                    ),
                )
            return cur.rowcount > 0

    def set_last_error(self, run_id: str, message: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE runs SET last_error = ?
                 WHERE id = ? AND status = 'running'
                """,
                (message[:4000], run_id),
            )

    def finish_cancelled(self, run_id: str, reason: str) -> bool:
        """Finalise a cancelled run; safe to call twice."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    """
                    UPDATE runs
                       SET status = 'cancelled',
                           finished_at = COALESCE(finished_at, ?),
                           phase = 'cancelled',
                           cancel_reason = COALESCE(cancel_reason, ?)
                     WHERE id = ? AND status = 'running'
                    """,
                    (utc_now_millis(), reason[:1000], run_id),
                )
                changed = cur.rowcount > 0
                self._delete_stage(run_id)
                if changed:
                    self._clear_active_if(run_id)
                self._conn.execute("COMMIT")
                return changed
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def finish_failed(self, run_id: str, error: str) -> bool:
        """Finalise a failed run.  Staging is discarded, published data not touched."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    """
                    UPDATE runs
                       SET status = 'failed',
                           finished_at = COALESCE(finished_at, ?),
                           phase = 'failed',
                           last_error = ?
                     WHERE id = ? AND status = 'running'
                    """,
                    (utc_now_millis(), error[:4000], run_id),
                )
                changed = cur.rowcount > 0
                self._delete_stage(run_id)
                if changed:
                    self._clear_active_if(run_id)
                self._conn.execute("COMMIT")
                return changed
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def reset_stage(self, run_id: str) -> bool:
        """Discard all pages collected for the current attempt."""
        with self._lock:
            if not self.is_run_running(run_id):
                return False
            self._conn.execute(
                "DELETE FROM run_records WHERE run_id = ?", (run_id,)
            )
            return True

    def stage_records(
        self, run_id: str, records: Sequence[dict[str, Any]], seq_start: int
    ) -> int:
        """Insert one page into staging, keeping one row per identifier.

        If the same identifier occurs again (whether because of a malformed
        source or a repeated page), the record with the newer OAI datestamp
        wins; the later observation wins a tie.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if not self.is_run_running(run_id):
                    self._conn.execute("ROLLBACK")
                    return 0
                for offset, record in enumerate(records):
                    seq = seq_start + offset
                    self._conn.execute(
                        """
                        INSERT INTO run_records
                            (run_id, identifier, title, creator, datestamp,
                             deleted, seq)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(run_id, identifier) DO UPDATE SET
                            title = excluded.title,
                            creator = excluded.creator,
                            datestamp = excluded.datestamp,
                            deleted = excluded.deleted,
                            seq = excluded.seq
                        WHERE excluded.datestamp > run_records.datestamp
                           OR (excluded.datestamp = run_records.datestamp
                               AND excluded.seq > run_records.seq)
                        """,
                        (
                            run_id,
                            record["identifier"],
                            record.get("title"),
                            record.get("creator"),
                            record["datestamp"],
                            1 if record.get("deleted") else 0,
                            seq,
                        ),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            return len(records)

    def _delete_stage(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM run_records WHERE run_id = ?", (run_id,))

    def _clear_active_if(self, run_id: str) -> None:
        if self._meta_get("active_run_id") == run_id:
            self._meta_set("active_run_id", None)

    def _reconcile_attributions(
        self, identifier: str, old_creator: str | None, new_creator: str | None
    ) -> None:
        """Drop assignments whose position/text did not survive an update.

        Called inside the commit transaction.  An assignment survives only
        when the same position holds exactly the same text before and after
        the harvest; a renamed, reordered or truncated author list releases
        the affected positions.  A brand-new record keeps nothing.
        """
        old_parts = split_creators(old_creator)
        new_parts = split_creators(new_creator)
        kept = {
            position
            for position, (old, new) in enumerate(zip(old_parts, new_parts))
            if old == new
        }
        rows = self._conn.execute(
            "SELECT position FROM attributions WHERE identifier = ?",
            (identifier,),
        ).fetchall()
        for row in rows:
            if row["position"] not in kept:
                self._conn.execute(
                    "DELETE FROM attributions "
                    "WHERE identifier = ? AND position = ?",
                    (identifier, row["position"]),
                )

    def commit_run(self, run_id: str) -> dict[str, Any]:
        """Atomically publish staging and advance the watermark.

        The check for ownership (``active_run_id``) happens inside the same
        IMMEDIATE transaction as the catalogue writes.  A cancelled or
        superseded round can therefore never publish, even if its HTTP
        response arrives late.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                run = self._conn.execute(
                    "SELECT * FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                active = self._meta_get("active_run_id")
                if (
                    run is None
                    or run["status"] != "running"
                    or active != run_id
                    or not run["window_until"]
                ):
                    self._conn.execute("ROLLBACK")
                    return {"committed": False, "reason": "run is not active"}

                rows = self._conn.execute(
                    """
                    SELECT identifier, title, creator, datestamp, deleted
                      FROM run_records WHERE run_id = ?
                     ORDER BY seq
                    """,
                    (run_id,),
                ).fetchall()

                now = utc_now_millis()
                upserts = 0
                deletions = 0
                for row in rows:
                    if row["deleted"]:
                        # 撤回：目录行与它的全部署名归属一起消失。
                        self._conn.execute(
                            "DELETE FROM attributions WHERE identifier = ?",
                            (row["identifier"],),
                        )
                        self._conn.execute(
                            "DELETE FROM catalog WHERE identifier = ?",
                            (row["identifier"],),
                        )
                        deletions += 1
                    else:
                        previous = self._conn.execute(
                            "SELECT creator FROM catalog WHERE identifier = ?",
                            (row["identifier"],),
                        ).fetchone()
                        self._conn.execute(
                            """
                            INSERT INTO catalog
                                (identifier, title, creator, source_datestamp,
                                 harvested_at)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(identifier) DO UPDATE SET
                                title = excluded.title,
                                creator = excluded.creator,
                                source_datestamp = excluded.source_datestamp,
                                harvested_at = excluded.harvested_at
                            """,
                            (
                                row["identifier"],
                                row["title"] or "",
                                row["creator"],
                                row["datestamp"],
                                now,
                            ),
                        )
                        # 作者顺序或姓名被本轮改动时，只保留序位与文字
                        # 都未变的归属；未变化的署名归属原样保留。
                        self._reconcile_attributions(
                            row["identifier"],
                            previous["creator"] if previous else None,
                            row["creator"],
                        )
                        upserts += 1

                watermark = run["window_until"]
                self._conn.execute(
                    """
                    UPDATE published_state
                       SET watermark = ?,
                           updated_at = ?,
                           success_count = success_count + 1,
                           last_run_id = ?
                     WHERE id = 1
                    """,
                    (watermark, now, run_id),
                )
                self._conn.execute(
                    """
                    UPDATE runs
                       SET status = 'succeeded',
                           finished_at = ?,
                           phase = 'published',
                           watermark_after = ?,
                           last_error = NULL
                     WHERE id = ?
                    """,
                    (now, watermark, run_id),
                )
                self._clear_active_if(run_id)
                self._delete_stage(run_id)
                self._conn.execute("COMMIT")
                return {
                    "committed": True,
                    "upserts": upserts,
                    "deletions": deletions,
                    "watermark": watermark,
                }
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def fail_run(self, run_id: str, error: str) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    """
                    UPDATE runs
                       SET status = 'failed',
                           finished_at = ?,
                           phase = 'failed',
                           last_error = ?
                     WHERE id = ? AND status = 'running'
                    """,
                    (utc_now_millis(), error[:4000], run_id),
                )
                self._delete_stage(run_id)
                self._clear_active_if(run_id)
                self._conn.execute("COMMIT")
                return cur.rowcount > 0
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def cancel_run(self, run_id: str, reason: str) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    """
                    UPDATE runs
                       SET status = 'cancelled',
                           finished_at = ?,
                           phase = 'cancelled',
                           cancel_reason = ?
                     WHERE id = ? AND status = 'running'
                    """,
                    (utc_now_millis(), reason[:1000], run_id),
                )
                self._delete_stage(run_id)
                self._clear_active_if(run_id)
                self._conn.execute("COMMIT")
                return cur.rowcount > 0
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    # --------------------------------------------------------------- published
    def published_state(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM published_state WHERE id = 1"
            ).fetchone()
            result = dict(row) if row else {
                "watermark": None,
                "updated_at": None,
                "success_count": 0,
                "last_run_id": None,
            }
            count = self._conn.execute(
                "SELECT COUNT(*) AS n FROM catalog"
            ).fetchone()
            result["catalog_size"] = int(count["n"])
            return result

    def catalog_page(
        self, query: str = "", page: int = 1, page_size: int = 25
    ) -> dict[str, Any]:
        query = (query or "").strip()
        page = max(1, int(page or 1))
        page_size = min(200, max(1, int(page_size or 25)))
        like = f"%{query}%"
        where = ""
        params: list[Any] = []
        if query:
            where = (
                "WHERE identifier LIKE ? OR title LIKE ? "
                "OR COALESCE(creator, '') LIKE ?"
            )
            params = [like, like, like]
        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM catalog {where}", params
            ).fetchone()["n"]
            total = int(total)
            pages = max(1, (total + page_size - 1) // page_size)
            if page > pages:
                page = pages
            rows = self._conn.execute(
                f"""
                SELECT identifier, title, creator, source_datestamp,
                       harvested_at
                  FROM catalog {where}
                 ORDER BY title COLLATE NOCASE, identifier
                 LIMIT ? OFFSET ?
                """,
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
            return {
                "records": [dict(r) for r in rows],
                "total": total,
                "page": page,
                "page_size": page_size,
                "pages": pages,
                "query": query,
            }

    # ---------------------------------------------------------------- people
    # 署名归并：把已发布书目的署名（按分号拆分后的某一序位）人工归到人物编号。
    # 版本令牌覆盖归并页展示的全部数据（目录、人物、归属），提交时逐项核对，
    # 过期提交一律拒绝。

    def _merge_version_locked(self) -> str:
        digest = hashlib.sha256()
        for row in self._conn.execute(
            "SELECT identifier, title, creator FROM catalog ORDER BY identifier"
        ):
            digest.update(
                "\x1f".join(
                    [row["identifier"], row["title"] or "", row["creator"] or ""]
                ).encode("utf-8")
            )
            digest.update(b"\x1e")
        for row in self._conn.execute(
            "SELECT id, display_name FROM persons ORDER BY id"
        ):
            digest.update(
                "\x1f".join([row["id"], row["display_name"]]).encode("utf-8")
            )
            digest.update(b"\x1e")
        for row in self._conn.execute(
            "SELECT identifier, position, name_text, person_id FROM attributions "
            "ORDER BY identifier, position"
        ):
            digest.update(
                "\x1f".join(
                    [
                        row["identifier"],
                        str(row["position"]),
                        row["name_text"],
                        row["person_id"],
                    ]
                ).encode("utf-8")
            )
            digest.update(b"\x1e")
        return digest.hexdigest()[:16]

    def merge_version(self) -> str:
        with self._lock:
            return self._merge_version_locked()

    def merge_state(self) -> dict[str, Any]:
        """Everything the merge page shows, plus the version token for it.

        Catalog, persons and assignments are read under the store lock, so a
        concurrent harvest commit is either fully visible or not at all: a
        person's bibliography always matches the same published catalogue.
        """
        with self._lock:
            persons = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT id, display_name, created_at FROM persons ORDER BY id"
                ).fetchall()
            ]
            names = {p["id"]: p["display_name"] for p in persons}
            assignments: dict[tuple[str, int], sqlite3.Row] = {}
            counts: dict[str, int] = {}
            for row in self._conn.execute(
                "SELECT identifier, position, person_id, name_text "
                "FROM attributions"
            ).fetchall():
                assignments[(row["identifier"], row["position"])] = row
                counts[row["person_id"]] = counts.get(row["person_id"], 0) + 1
            bibliographies: dict[str, list[dict[str, Any]]] = {}
            for row in self._conn.execute(
                """
                SELECT DISTINCT a.person_id AS person_id,
                       c.identifier AS identifier,
                       c.title AS title,
                       c.source_datestamp AS source_datestamp
                  FROM attributions a
                  JOIN catalog c ON c.identifier = a.identifier
                 ORDER BY c.title COLLATE NOCASE, c.identifier
                """
            ).fetchall():
                bibliographies.setdefault(row["person_id"], []).append(
                    {
                        "identifier": row["identifier"],
                        "title": row["title"],
                        "source_datestamp": row["source_datestamp"],
                    }
                )
            attributions: list[dict[str, Any]] = []
            for rec in self._conn.execute(
                "SELECT identifier, title, creator, source_datestamp FROM catalog "
                "ORDER BY title COLLATE NOCASE, identifier"
            ).fetchall():
                for position, name in enumerate(split_creators(rec["creator"])):
                    assignment = assignments.get((rec["identifier"], position))
                    attributions.append(
                        {
                            "identifier": rec["identifier"],
                            "title": rec["title"],
                            "source_datestamp": rec["source_datestamp"],
                            "position": position,
                            "name_text": name,
                            "person_id": (
                                assignment["person_id"] if assignment else None
                            ),
                            "person_name": (
                                names.get(assignment["person_id"])
                                if assignment
                                else None
                            ),
                        }
                    )
            for person in persons:
                person["attribution_count"] = counts.get(person["id"], 0)
                person["bibliography"] = bibliographies.get(person["id"], [])
            return {
                "version": self._merge_version_locked(),
                "attributions": attributions,
                "persons": persons,
                "unassigned": sum(
                    1 for item in attributions if item["person_id"] is None
                ),
            }

    def _next_person_id_locked(self) -> str:
        current = self._meta_get("person_seq")
        number = int(current or "0") + 1
        self._meta_set("person_seq", str(number))
        return f"P{number:06d}"

    def assign_attribution(
        self,
        *,
        identifier: str,
        position: int,
        expected_version: str,
        person_id: str | None = None,
        new_person_name: str | None = None,
    ) -> dict[str, Any]:
        """Attach one published attribution to a person (creating it if asked).

        The whole check-and-set runs in a single IMMEDIATE transaction: the
        ``expected_version`` must still describe what the user saw, the
        attribution must exist in the published catalogue, and it must still
        be unassigned (one attribution belongs to at most one person).  No
        automatic merging by name ever happens here.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if self._merge_version_locked() != expected_version:
                    self._conn.execute("ROLLBACK")
                    return {"ok": False, "reason": "stale"}
                record = self._conn.execute(
                    "SELECT creator FROM catalog WHERE identifier = ?",
                    (identifier,),
                ).fetchone()
                parts = split_creators(record["creator"]) if record else []
                if record is None or not 0 <= position < len(parts):
                    self._conn.execute("ROLLBACK")
                    return {"ok": False, "reason": "missing"}
                existing = self._conn.execute(
                    "SELECT person_id FROM attributions "
                    "WHERE identifier = ? AND position = ?",
                    (identifier, position),
                ).fetchone()
                if existing is not None:
                    self._conn.execute("ROLLBACK")
                    return {
                        "ok": False,
                        "reason": "conflict",
                        "person_id": existing["person_id"],
                    }
                created = False
                if person_id is None:
                    display = (
                        (new_person_name or "").strip() or parts[position]
                    )[:200]
                    person_id = self._next_person_id_locked()
                    self._conn.execute(
                        "INSERT INTO persons (id, display_name, created_at) "
                        "VALUES (?, ?, ?)",
                        (person_id, display, utc_now_millis()),
                    )
                    created = True
                elif (
                    self._conn.execute(
                        "SELECT 1 FROM persons WHERE id = ?", (person_id,)
                    ).fetchone()
                    is None
                ):
                    self._conn.execute("ROLLBACK")
                    return {"ok": False, "reason": "no_person"}
                self._conn.execute(
                    "INSERT INTO attributions "
                    "(identifier, position, name_text, person_id, assigned_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        identifier,
                        position,
                        parts[position],
                        person_id,
                        utc_now_millis(),
                    ),
                )
                self._conn.execute("COMMIT")
                return {
                    "ok": True,
                    "person_id": person_id,
                    "created_person": created,
                    "name_text": parts[position],
                }
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def unassign_attribution(
        self, *, identifier: str, position: int, expected_version: str
    ) -> dict[str, Any]:
        """Release one attribution from its person, with the same staleness
        check as :meth:`assign_attribution`."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if self._merge_version_locked() != expected_version:
                    self._conn.execute("ROLLBACK")
                    return {"ok": False, "reason": "stale"}
                existing = self._conn.execute(
                    "SELECT person_id FROM attributions "
                    "WHERE identifier = ? AND position = ?",
                    (identifier, position),
                ).fetchone()
                if existing is None:
                    self._conn.execute("ROLLBACK")
                    return {"ok": False, "reason": "conflict"}
                self._conn.execute(
                    "DELETE FROM attributions "
                    "WHERE identifier = ? AND position = ?",
                    (identifier, position),
                )
                self._conn.execute("COMMIT")
                return {"ok": True, "person_id": existing["person_id"]}
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()
