"""
StatLLM Database Manager.

Maintains sample archives and weighted multi-token / sequence-trait frequency matrices.
"""

import sqlite3
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


class Database:
    def __init__(self, db_path: str = "statllm.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables and indices."""
        with self._get_connection() as conn:
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

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

            conn.commit()

    def add_model(self, name: str, display_name: str, provider: str = "", color: str = "#3b82f6"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO models (name, provider, display_name, color)
            VALUES (?, ?, ?, ?)
            """, (name, provider, display_name, color))
            conn.commit()

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
        weight: float = 1.0
    ) -> int:
        """
        Adds a sample and immediately updates token_counts.
        """
        parsed_json = json.dumps(parsed_tokens, ensure_ascii=False)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO samples 
            (model_name, probe_id, raw_text, parsed_value, is_valid, strictly_complied, source_type, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                probe_id,
                raw_text,
                parsed_json,
                1 if is_valid else 0,
                1 if strictly_complied else 0,
                source_type,
                float(weight)
            ))
            sample_id = cursor.lastrowid

            # Update token counts for each token in array
            for tok in parsed_tokens:
                cursor.execute("""
                INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                weighted_count = weighted_count + ?
                """, (model_name, probe_id, tok, float(weight), float(weight)))

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

                if "canonical_perm" in traits and traits["canonical_perm"] != "INVALID":
                    perm_tok = f"perm:{traits['canonical_perm']}"
                    cursor.execute("""
                    INSERT INTO token_counts (model_name, probe_id, token, weighted_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_name, probe_id, token) DO UPDATE SET
                    weighted_count = weighted_count + ?
                    """, (model_name, probe_id, perm_tok, float(weight), float(weight)))

            conn.commit()
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
                for t in tokens:
                    token_deltas[(m, pid, str(t))] += w
                
                traits = s.get("traits", {})
                if "has_duplicates" in traits:
                    token_deltas[(m, pid, f"trait:has_dup_{traits['has_duplicates']}")] += w
                if "is_sorted" in traits:
                    token_deltas[(m, pid, f"trait:is_sorted_{traits['is_sorted']}")] += w
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
                # Only count primary element tokens (exclude trait: tokens from total denominator)
                if not tok.startswith("trait:") and not tok.startswith("perm:"):
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

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics of database."""
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

            return {
                "total_samples": total_samples,
                "official_samples": official_samples,
                "user_samples": user_samples,
                "model_breakdown": model_breakdown,
                "probe_breakdown": probe_breakdown
            }
