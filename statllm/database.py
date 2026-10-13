"""
StatLLM Database Manager.

Maintains sample archives and weighted multi-token / sequence-trait frequency matrices.
"""

import os
import sqlite3
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


class Database:
    def __init__(self, db_path: str = "statllm.db"):
        self.db_path = db_path
        self._cached_stats: Optional[Dict[str, Any]] = None
        self._cached_token_usage: Optional[Dict[str, Any]] = None
        self._cached_stats_sig: Optional[Tuple[int, int, int, int]] = None
        self._cached_token_usage_sig: Optional[Tuple[int, int, int, int]] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        """Initializes tables and indices."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            cursor = conn.cursor()
            
            # Models table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                name TEXT PRIMARY KEY,
                provider TEXT,
                display_name TEXT,
                color TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Raw samples table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                probe_id TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                parsed_value TEXT NOT NULL,       -- JSON serialized array
                is_valid INTEGER NOT NULL DEFAULT 1,
                strictly_complied INTEGER NOT NULL DEFAULT 1,
                source_type TEXT NOT NULL DEFAULT 'official',  -- 'official' or 'user'
                weight REAL NOT NULL DEFAULT 1.0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                perturbation_type TEXT NOT NULL DEFAULT 'none',
                perturbation_prefix TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Ensure new columns exist on existing databases
            cursor.execute("PRAGMA table_info(samples)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            for col_name, col_type in [
                ("prompt_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("completion_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("total_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("perturbation_type", "TEXT NOT NULL DEFAULT 'none'"),
                ("perturbation_prefix", "TEXT NOT NULL DEFAULT ''"),
                ("temperature", "REAL NOT NULL DEFAULT 0.8")
            ]:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE samples ADD COLUMN {col_name} {col_type};")

            # Pre-aggregated token frequency table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_counts (
                model_name TEXT NOT NULL,
                probe_id TEXT NOT NULL,
                token TEXT NOT NULL,
                weighted_count REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (model_name, probe_id, token)
            );
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_model_probe 
            ON samples (model_name, probe_id);
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS probe_matrices (
                probe_id TEXT PRIMARY KEY,
                matrix_json TEXT NOT NULL,
                mdi REAL NOT NULL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            conn.commit()

    def _get_db_signature(self) -> Tuple[int, int, int, int]:
        """Returns physical disk state signature of DB and WAL files to detect cross-process writes."""
        try:
            st1 = os.stat(self.db_path)
            wal_path = self.db_path + "-wal"
            st2 = os.stat(wal_path) if os.path.exists(wal_path) else None
            return (
                st1.st_mtime_ns,
                st1.st_size,
                st2.st_size if st2 else 0,
                st2.st_mtime_ns if st2 else 0
            )
        except Exception:
            return (0, 0, 0, 0)

    def invalidate_stats_cache(self):
        """Invalidates in-memory cached statistics."""
        self._cached_stats = None
        self._cached_token_usage = None
        self._cached_stats_sig = None
        self._cached_token_usage_sig = None

    def add_model(self, name: str, display_name: str, provider: str = "", color: str = "#3b82f6"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO models (name, provider, display_name, color)
            VALUES (?, ?, ?, ?)
            """, (name, provider, display_name, color))
            conn.commit()
        self.invalidate_stats_cache()

    def list_models(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models ORDER BY name")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def add_sample(
        self,
        model_name: str,
        probe_id: str,
        raw_text: str,
        parsed_tokens: List[str],
        traits: Dict[str, Any],
        is_valid: bool = True,
        strictly_complied: bool = True,
        source_type: str = "official",
        weight: float = 1.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        perturbation_type: str = "none",
        perturbation_prefix: str = "",
        temperature: float = 0.8
    ) -> int:
        """
        Adds a sample and immediately updates token_counts.
        """
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens
        parsed_json = json.dumps(parsed_tokens, ensure_ascii=False)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO samples 
            (model_name, probe_id, raw_text, parsed_value, is_valid, strictly_complied, source_type, weight,
             prompt_tokens, completion_tokens, total_tokens, perturbation_type, perturbation_prefix, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                probe_id,
                raw_text,
                parsed_json,
                1 if is_valid else 0,
                1 if strictly_complied else 0,
                source_type,
                float(weight),
                int(prompt_tokens),
                int(completion_tokens),
                int(total_tokens),
                perturbation_type,
                perturbation_prefix,
                float(temperature)
            ))
            sample_id = cursor.lastrowid

            # Update token counts for each token in array
            for idx, tok in enumerate(parsed_tokens):
                tok_str = str(tok)
                cursor.execute("""
                INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                weighted_count = weighted_count + ?
                """, (model_name, probe_id, tok_str, float(weight), float(weight)))

                pos_tok = f"pos:{idx}:{tok_str}"
                cursor.execute("""
                INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                weighted_count = weighted_count + ?
                """, (model_name, probe_id, pos_tok, float(weight), float(weight)))

                # Sub-token features for arr_int_big5
                if probe_id == "arr_int_big5":
                    if len(tok_str) >= 2:
                        pref_tok = f"pref:{tok_str[:2]}"
                        pos_pref_tok = f"pos_pref:{idx}:{tok_str[:2]}"
                        for ptok in (pref_tok, pos_pref_tok):
                            cursor.execute("""
                            INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                            weighted_count = weighted_count + ?
                            """, (model_name, probe_id, ptok, float(weight), float(weight)))
                    if len(tok_str) >= 1:
                        lead_tok = f"lead:{tok_str[0]}"
                        cursor.execute("""
                        INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                        weighted_count = weighted_count + ?
                        """, (model_name, probe_id, lead_tok, float(weight), float(weight)))

                # Sub-token phonotactic features for arr_pseudo_cvc5
                if probe_id == "arr_pseudo_cvc5" and len(tok_str) == 3:
                    for feat_tok in (f"init:{tok_str[0]}", f"vow:{tok_str[1]}", f"coda:{tok_str[2]}"):
                        cursor.execute("""
                        INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                        weighted_count = weighted_count + ?
                        """, (model_name, probe_id, feat_tok, float(weight), float(weight)))

            # Update sequence trait counts
            if traits:
                if "has_duplicates" in traits:
                    dup_tok = f"trait:has_dup_{traits['has_duplicates']}"
                    cursor.execute("""
                    INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                    weighted_count = weighted_count + ?
                    """, (model_name, probe_id, dup_tok, float(weight), float(weight)))

                if "is_sorted" in traits:
                    sort_tok = f"trait:is_sorted_{traits['is_sorted']}"
                    cursor.execute("""
                    INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                    weighted_count = weighted_count + ?
                    """, (model_name, probe_id, sort_tok, float(weight), float(weight)))

                if "has_consecutive" in traits:
                    consec_tok = f"trait:consec_{traits['has_consecutive']}"
                    cursor.execute("""
                    INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                    weighted_count = weighted_count + ?
                    """, (model_name, probe_id, consec_tok, float(weight), float(weight)))

                if "canonical_perm" in traits and traits["canonical_perm"] != "INVALID":
                    perm_tok = f"perm:{traits['canonical_perm']}"
                    cursor.execute("""
                    INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                    weighted_count = weighted_count + ?
                    """, (model_name, probe_id, perm_tok, float(weight), float(weight)))

            conn.commit()
            self.invalidate_stats_cache()
            return sample_id

    def add_samples_batch(self, samples_data: List[Dict[str, Any]]):
        """Batch insert samples and aggregated token counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert samples
            cursor.executemany("""
            INSERT INTO samples 
            (model_name, probe_id, raw_text, parsed_value, is_valid, strictly_complied, source_type, weight)
            VALUES (:model_name, :probe_id, :raw_text, :parsed_value, :is_valid, :strictly_complied, :source_type, :weight)
            """, [
                {
                    "model_name": s["model_name"],
                    "probe_id": s["probe_id"],
                    "raw_text": s["raw_text"],
                    "parsed_value": json.dumps(s["parsed_tokens"], ensure_ascii=False) if isinstance(s.get("parsed_tokens"), list) else str(s.get("parsed_value", "[]")),
                    "is_valid": 1 if s.get("is_valid", True) else 0,
                    "strictly_complied": 1 if s.get("strictly_complied", True) else 0,
                    "source_type": s.get("source_type", "official"),
                    "weight": float(s.get("weight", 1.0))
                }
                for s in samples_data
            ])

            # Accumulate token counts
            token_deltas = defaultdict(float)
            for s in samples_data:
                m = s["model_name"]
                pid = s["probe_id"]
                w = float(s.get("weight", 1.0))
                tokens = s.get("parsed_tokens", [])
                for idx, t in enumerate(tokens):
                    tok_str = str(t)
                    token_deltas[(m, pid, tok_str)] += w
                    token_deltas[(m, pid, f"pos:{idx}:{tok_str}")] += w
                    if pid == "arr_int_big5":
                        if len(tok_str) >= 2:
                            token_deltas[(m, pid, f"pref:{tok_str[:2]}")] += w
                            token_deltas[(m, pid, f"pos_pref:{idx}:{tok_str[:2]}")] += w
                        if len(tok_str) >= 1:
                            token_deltas[(m, pid, f"lead:{tok_str[0]}")] += w
                    elif pid == "arr_pseudo_cvc5" and len(tok_str) == 3:
                        token_deltas[(m, pid, f"init:{tok_str[0]}")] += w
                        token_deltas[(m, pid, f"vow:{tok_str[1]}")] += w
                        token_deltas[(m, pid, f"coda:{tok_str[2]}")] += w
                
                traits = s.get("traits", {})
                if "has_duplicates" in traits:
                    token_deltas[(m, pid, f"trait:has_dup_{traits['has_duplicates']}")] += w
                if "is_sorted" in traits:
                    token_deltas[(m, pid, f"trait:is_sorted_{traits['is_sorted']}")] += w
                if "has_consecutive" in traits:
                    token_deltas[(m, pid, f"trait:consec_{traits['has_consecutive']}")] += w
                if "canonical_perm" in traits and traits["canonical_perm"] != "INVALID":
                    token_deltas[(m, pid, f"perm:{traits['canonical_perm']}")] += w

            for (m, pid, tok), w in token_deltas.items():
                cursor.execute("""
                INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                weighted_count = weighted_count + ?
                """, (m, pid, tok, w, w))

            conn.commit()
            self.invalidate_stats_cache()

    def rebuild_token_counts(self):
        """Rebuilds token_counts table from all valid samples in database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM token_counts")
            cursor.execute("SELECT model_name, probe_id, parsed_value, weight FROM samples WHERE is_valid = 1")
            rows = cursor.fetchall()
            token_deltas = defaultdict(float)
            for row in rows:
                m = row["model_name"]
                pid = row["probe_id"]
                w = float(row["weight"])
                try:
                    tokens = json.loads(row["parsed_value"])
                except Exception:
                    continue
                for idx, tok in enumerate(tokens):
                    tok_str = str(tok)
                    token_deltas[(m, pid, tok_str)] += w
                    token_deltas[(m, pid, f"pos:{idx}:{tok_str}")] += w
                    if pid == "arr_int_big5":
                        if len(tok_str) >= 2:
                            token_deltas[(m, pid, f"pref:{tok_str[:2]}")] += w
                            token_deltas[(m, pid, f"pos_pref:{idx}:{tok_str[:2]}")] += w
                        if len(tok_str) >= 1:
                            token_deltas[(m, pid, f"lead:{tok_str[0]}")] += w
                    elif pid == "arr_pseudo_cvc5" and len(tok_str) == 3:
                        token_deltas[(m, pid, f"init:{tok_str[0]}")] += w
                        token_deltas[(m, pid, f"vow:{tok_str[1]}")] += w
                        token_deltas[(m, pid, f"coda:{tok_str[2]}")] += w

                has_dup = len(tokens) != len(set(tokens))
                token_deltas[(m, pid, f"trait:has_dup_{has_dup}")] += w
                if pid == "arr_int5":
                    try:
                        nums = [int(x) for x in tokens]
                        is_sorted = nums == sorted(nums)
                        token_deltas[(m, pid, f"trait:is_sorted_{is_sorted}")] += w
                    except Exception:
                        pass
                elif pid == "arr_int_big5":
                    try:
                        nums = [int(x) for x in tokens]
                        is_sorted = (nums == sorted(nums)) or (nums == sorted(nums, reverse=True))
                        token_deltas[(m, pid, f"trait:is_sorted_{is_sorted}")] += w
                        has_consec = False
                        for t in tokens:
                            st = str(t)
                            if len(st) >= 4:
                                diffs = [int(st[i+1]) - int(st[i]) for i in range(len(st)-1)]
                                if all(d == 1 for d in diffs) or all(d == -1 for d in diffs):
                                    has_consec = True
                                    break
                        token_deltas[(m, pid, f"trait:consec_{has_consec}")] += w
                    except Exception:
                        pass
                elif pid == "arr_perm5":
                    token_deltas[(m, pid, f"perm:{','.join(str(x) for x in tokens)}")] += w

            for (m, pid, tok), w in token_deltas.items():
                cursor.execute("""
                INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                weighted_count = weighted_count + ?
                """, (m, pid, tok, w, w))
            conn.commit()
            self.invalidate_stats_cache()

    def get_sample_count(self, model_name: str, probe_id: Optional[str] = None) -> int:
        """Returns the number of samples for a model and optional probe."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if probe_id:
                cursor.execute(
                    "SELECT count(*) FROM samples WHERE model_name = ? AND probe_id = ?",
                    (model_name, probe_id)
                )
            else:
                cursor.execute(
                    "SELECT count(*) FROM samples WHERE model_name = ?",
                    (model_name,)
                )
            return cursor.fetchone()[0]

    def get_all_model_probe_counts(self) -> Dict[str, Dict[str, Tuple[Dict[str, float], float]]]:
        """
        Loads the pre-aggregated token count matrix:
        res[model_name][probe_id] = (counts_dict, total_weight)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT model_name, probe_id, token, weighted_count
            FROM token_counts
            """)
            
            res = defaultdict(lambda: defaultdict(lambda: [defaultdict(float), 0.0]))
            for row in cursor.fetchall():
                m = row["model_name"]
                p = row["probe_id"]
                tok = row["token"]
                w = float(row["weighted_count"])
                # Only count primary element tokens (exclude trait:, perm:, pos: tokens from total denominator)
                if not tok.startswith("trait:") and not tok.startswith("perm:") and not tok.startswith("pos:"):
                    res[m][p][0][tok] = w
                    res[m][p][1] += w
                else:
                    res[m][p][0][tok] = w

            final_res = {}
            for m, probes in res.items():
                final_res[m] = {}
                for p, (c_dict, tot) in probes.items():
                    final_res[m][p] = (dict(c_dict), float(tot))
            return final_res

    def clear_database(self):
        """Clears all data for reseeding."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM samples")
            cursor.execute("DELETE FROM token_counts")
            cursor.execute("DELETE FROM models")
            conn.commit()
            self.invalidate_stats_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics of database (cached in-memory, auto-refreshed on db change)."""
        sig = self._get_db_signature()
        if self._cached_stats is not None and getattr(self, "_cached_stats_sig", None) == sig:
            return self._cached_stats

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM samples")
            total_samples = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM samples WHERE source_type = 'official'")
            official_samples = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM samples WHERE source_type = 'user'")
            user_samples = cursor.fetchone()[0]

            cursor.execute("""
            SELECT model_name, COUNT(*) as count, SUM(weight) as weighted_count
            FROM samples
            GROUP BY model_name
            ORDER BY count DESC
            """)
            model_breakdown = [dict(r) for r in cursor.fetchall()]

            cursor.execute("""
            SELECT probe_id, COUNT(*) as count
            FROM samples
            GROUP BY probe_id
            """)
            probe_breakdown = [dict(r) for r in cursor.fetchall()]

            # Token usage summary
            cursor.execute("""
            SELECT 
                COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                COALESCE(SUM(total_tokens), 0) as grand_total_tokens,
                COUNT(CASE WHEN total_tokens > 0 THEN 1 END) as recorded_api_calls
            FROM samples
            """)
            token_usage = dict(cursor.fetchone())

            cursor.execute("""
            SELECT perturbation_type, COUNT(*) as count
            FROM samples
            GROUP BY perturbation_type
            ORDER BY count DESC
            """)
            perturbation_breakdown = [dict(r) for r in cursor.fetchall()]

            res = {
                "total_samples": total_samples,
                "official_samples": official_samples,
                "user_samples": user_samples,
                "model_breakdown": model_breakdown,
                "probe_breakdown": probe_breakdown,
                "token_usage": token_usage,
                "perturbation_breakdown": perturbation_breakdown
            }
            self._cached_stats = res
            self._cached_stats_sig = sig
            return res

    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Returns detailed breakdown of tokens consumed by model and perturbation (cached, auto-refreshed)."""
        sig = self._get_db_signature()
        if self._cached_token_usage is not None and getattr(self, "_cached_token_usage_sig", None) == sig:
            return self._cached_token_usage

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT 
                COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                COALESCE(SUM(total_tokens), 0) as grand_total_tokens,
                COUNT(CASE WHEN total_tokens > 0 THEN 1 END) as recorded_api_calls,
                COUNT(*) as total_samples
            FROM samples
            """)
            overall = dict(cursor.fetchone())

            cursor.execute("""
            SELECT 
                model_name,
                COUNT(*) as sample_count,
                COUNT(CASE WHEN total_tokens > 0 THEN 1 END) as api_call_count,
                COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) as completion_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens
            FROM samples
            GROUP BY model_name
            ORDER BY total_tokens DESC
            """)
            by_model = [dict(r) for r in cursor.fetchall()]

            cursor.execute("""
            SELECT 
                perturbation_type,
                COUNT(*) as sample_count,
                COALESCE(SUM(total_tokens), 0) as total_tokens
            FROM samples
            GROUP BY perturbation_type
            ORDER BY sample_count DESC
            """)
            by_perturbation = [dict(r) for r in cursor.fetchall()]

            res = {
                "overall": overall,
                "by_model": by_model,
                "by_perturbation": by_perturbation
            }
            self._cached_token_usage = res
            self._cached_token_usage_sig = sig
            return res

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Returns all sample records ordered by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM samples ORDER BY id ASC")
            return [dict(r) for r in cursor.fetchall()]

    def clear_all_data(self):
        """Wipes samples and token_counts tables while preserving schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM samples")
            cursor.execute("DELETE FROM token_counts")
            conn.commit()

    def get_probe_matrix_cache(self, probe_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves precomputed matrix dictionary for a specific probe."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT matrix_json FROM probe_matrices WHERE probe_id = ?", (probe_id,))
            row = cursor.fetchone()
            if row and row["matrix_json"]:
                try:
                    return json.loads(row["matrix_json"])
                except Exception:
                    return None
            return None

    def set_probe_matrix_cache(self, probe_id: str, matrix_dict: Dict[str, Any], mdi: float = 0.0):
        """Saves or updates precomputed matrix dictionary for a specific probe."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO probe_matrices (probe_id, matrix_json, mdi, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (probe_id, json.dumps(matrix_dict), float(mdi)))
            conn.commit()

    def get_all_probe_matrices_cache(self) -> Dict[str, Dict[str, Any]]:
        """Retrieves all precomputed probe matrices in a single fast query."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT probe_id, matrix_json FROM probe_matrices")
            rows = cursor.fetchall()
            result = {}
            for r in rows:
                try:
                    result[r["probe_id"]] = json.loads(r["matrix_json"])
                except Exception:
                    continue
            return result
