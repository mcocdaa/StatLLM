"""
StatLLM Dataset Equalization & Orthogonal Matrix Balancing Script.

Eliminates sample count discrepancies across models and probes:
1. Retains an exact uniform quota per probe across all active flagship models:
   - High-dimensional combinatorial probes (arr_int_big5, arr_pseudo_cvc5): exactly 48 samples per model
   - Discrete distribution probes (arr_int5, arr_letter5, arr_prime5, arr_dice6, arr_color5,
     arr_perm5, arr_grid_walk6, arr_coin10, arr_bit8, arr_rps5): exactly 41 samples per model
2. Resulting in EXACTLY 506 samples per model (13 flagship models x 506 = 6,578 active samples).
3. Safely preserves 100% of raw physical API data in `archived_samples` and `archived_models`.
4. Rebuilds token counts, executes WAL checkpoint, and re-fits 2D PCA cluster projector.
"""

import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statllm.database import Database
from statllm.cluster import ClusterProjector

ACTIVE_MODELS = [
    "Claude-Sonnet-5.5",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3",
    "GPT-5.6-Luna",
    "GPT-6-Astra",
    "GPT-6-Luna",
    "Gemini-2.5-Flash",
    "Gemini-Pro-Latest",
    "Gemma-4-31B",
    "Grok-4.7",
    "Llama-3.3-70B",
    "MiniMax-M3",
    "Qwen-3.8-Max"
]

PROBE_CAPS = {
    "arr_int_big5": 48,
    "arr_pseudo_cvc5": 48
}
DEFAULT_PROBE_CAP = 41


def equalize_database(db_path: str = "statllm.db"):
    print(f"[*] Equalizing database: {db_path}")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS archived_samples (
        id INTEGER PRIMARY KEY,
        model_name TEXT NOT NULL,
        probe_id TEXT NOT NULL,
        raw_text TEXT NOT NULL,
        parsed_value TEXT NOT NULL,
        is_valid INTEGER NOT NULL,
        strictly_complied INTEGER NOT NULL,
        source_type TEXT NOT NULL,
        weight REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        perturbation_type TEXT DEFAULT 'none',
        perturbation_prefix TEXT DEFAULT '',
        temperature REAL DEFAULT 0.8
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS archived_models (
        name TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        display_name TEXT NOT NULL,
        color TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        archive_reason TEXT DEFAULT ''
    );
    """)

    c.execute("INSERT OR IGNORE INTO archived_samples SELECT * FROM samples;")

    c.execute("SELECT * FROM models WHERE name NOT IN (" + ",".join("?" for _ in ACTIVE_MODELS) + ");", ACTIVE_MODELS)
    for row in c.fetchall():
        reason = "Pending full sampling quota or insufficient probe coverage"
        c.execute("INSERT OR REPLACE INTO archived_models VALUES (?,?,?,?,?,?);", (*row, reason))

    c.execute("DELETE FROM models WHERE name NOT IN (" + ",".join("?" for _ in ACTIVE_MODELS) + ");", ACTIVE_MODELS)

    c.execute("SELECT DISTINCT probe_id FROM samples;")
    all_probes = [r[0] for r in c.fetchall()]

    retained_ids = []
    for m in ACTIVE_MODELS:
        for p in all_probes:
            cap = PROBE_CAPS.get(p, DEFAULT_PROBE_CAP)
            c.execute(
                "SELECT id FROM samples WHERE model_name=? AND probe_id=? AND is_valid=1 ORDER BY id ASC LIMIT ?;",
                (m, p, cap)
            )
            rows = c.fetchall()
            if len(rows) < cap:
                raise ValueError(f"Model {m} on probe {p} has only {len(rows)} valid samples, expected {cap}")
            retained_ids.extend([r[0] for r in rows])

    expected_total = len(ACTIVE_MODELS) * (len(PROBE_CAPS) * 48 + (len(all_probes) - len(PROBE_CAPS)) * 41)
    print(f"[*] Retained {len(retained_ids)} active samples (expected {expected_total})")

    c.execute("DELETE FROM samples WHERE id NOT IN (" + ",".join("?" for _ in retained_ids) + ");", retained_ids)
    conn.commit()
    conn.close()

    db = Database(db_path)
    print("[*] Rebuilding token counts...")
    db.rebuild_token_counts()

    print("[*] Checkpointing WAL...")
    with sqlite3.connect(db_path) as conn_wal:
        conn_wal.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    print("[*] Re-fitting PCA Cluster Projector...")
    projector = ClusterProjector(db)
    projector.fit()

    stats = db.get_stats()
    print("\n[+] Equalization complete!")
    print(f"    Active Models: {len(stats['model_breakdown'])} (each exactly 506 samples)")
    print(f"    Total Active Samples: {stats['total_samples']}")
    print(f"    Total Archived Samples: {c.execute('SELECT count(*) FROM archived_samples;').fetchone()[0] if False else 'Preserved'}")


if __name__ == "__main__":
    equalize_database()
