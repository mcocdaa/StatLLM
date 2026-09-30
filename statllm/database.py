"""
StatLLM Database Manager.

Handles SQLite storage for ground-truth benchmark distributions,
crowdsourced sample collection with custom weights, and fast aggregated queries.
"""

import sqlite3
import json
import os
from typing import Dict, Any, List, Optional, Tuple
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

            # Samples table (raw text, parsed token, source_type, weight)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                probe_id TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                parsed_value TEXT NOT NULL,
                is_valid INTEGER NOT NULL DEFAULT 1,
                strictly_complied INTEGER NOT NULL DEFAULT 1,
                source_type TEXT NOT NULL DEFAULT 'official',  -- 'official' or 'user'
                weight REAL NOT NULL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Fast index on (model_name, probe_id, parsed_value)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_query 
            ON samples (model_name, probe_id, parsed_value);
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_source
            ON samples (source_type);
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
        parsed_value: str,
        is_valid: bool = True,
        strictly_complied: bool = True,
        source_type: str = "official",
        weight: float = 1.0
    ) -> int:
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
                parsed_value,
                1 if is_valid else 0,
                1 if strictly_complied else 0,
                source_type,
                float(weight)
            ))
            conn.commit()
            return cursor.lastrowid

    def add_samples_batch(self, samples_data: List[Dict[str, Any]]):
        """Batch insert samples for efficiency."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
            INSERT INTO samples 
            (model_name, probe_id, raw_text, parsed_value, is_valid, strictly_complied, source_type, weight)
            VALUES (:model_name, :probe_id, :raw_text, :parsed_value, :is_valid, :strictly_complied, :source_type, :weight)
            """, [
                {
                    "model_name": s["model_name"],
                    "probe_id": s["probe_id"],
                    "raw_text": s["raw_text"],
                    "parsed_value": s["parsed_value"],
                    "is_valid": 1 if s.get("is_valid", True) else 0,
                    "strictly_complied": 1 if s.get("strictly_complied", True) else 0,
                    "source_type": s.get("source_type", "official"),
                    "weight": float(s.get("weight", 1.0))
                }
                for s in samples_data
            ])
            conn.commit()

    def get_weighted_counts(self, model_name: str, probe_id: str) -> Tuple[Dict[str, float], float]:
        """
        Returns (counts_dict, total_weight) for a given model and probe.
        Counts are weighted: sum(weight) per parsed_value.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT parsed_value, SUM(weight) as total_weight
            FROM samples
            WHERE model_name = ? AND probe_id = ?
            GROUP BY parsed_value
            """, (model_name, probe_id))
            
            counts = {}
            total = 0.0
            for row in cursor.fetchall():
                w = float(row["total_weight"])
                counts[row["parsed_value"]] = w
                total += w
            return counts, total

    def get_all_model_probe_counts(self) -> Dict[str, Dict[str, Tuple[Dict[str, float], float]]]:
        """
        Loads the complete pre-aggregated count matrix in one fast query:
        result[model_name][probe_id] = (counts_dict, total_weight)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT model_name, probe_id, parsed_value, SUM(weight) as total_weight
            FROM samples
            GROUP BY model_name, probe_id, parsed_value
            """)
            
            res = defaultdict(lambda: defaultdict(lambda: [defaultdict(float), 0.0]))
            for row in cursor.fetchall():
                m = row["model_name"]
                p = row["probe_id"]
                val = row["parsed_value"]
                w = float(row["total_weight"])
                res[m][p][0][val] = w
                res[m][p][1] += w

            # Convert to standard dict of tuples
            final_res = {}
            for m, probes in res.items():
                final_res[m] = {}
                for p, (c_dict, tot) in probes.items():
                    final_res[m][p] = (dict(c_dict), float(tot))
            return final_res

    def get_stats(self) -> Dict[str, Any]:
        """Returns overview statistics of dataset."""
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
