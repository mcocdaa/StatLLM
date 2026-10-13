"""
Precompute out-of-sample pairwise matrices for all 12 probes in StatLLM.
Saves results permanently into statllm.db (probe_matrices table).
Reduces Single-Probe Lab and probe ranking latency from 140s to < 2ms.
"""

import time
import sqlite3
from statllm.database import Database
from statllm.engine import LikelihoodEvaluator
from statllm.matrix import get_probe_matrix, rank_probes_by_discrimination
from statllm.probes import ALL_PROBES


def main():
    print("=" * 60)
    print("StatLLM Matrix Precomputation Engine")
    print(f"Target Probes: {len(ALL_PROBES)} discrete probes")
    print("=" * 60)

    db = Database("statllm.db")
    evaluator = LikelihoodEvaluator(db)

    total_start = time.time()
    for idx, pid in enumerate(ALL_PROBES.keys(), 1):
        t0 = time.time()
        print(f"[{idx}/{len(ALL_PROBES)}] Computing matrix for {pid} ...", end="", flush=True)
        m = get_probe_matrix(db, evaluator, pid, force_refresh=True)
        t_elapsed = time.time() - t0
        print(f" Done in {t_elapsed:.2f}s (MDI: {m['mdi']}, Tier: {m['tier']}, WinRate: {m['overall_win_rate']*100:.1f}%)")

    # Flush WAL into main database file
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    # Verification: measure fast cached ranking retrieval
    t_rank_0 = time.time()
    ranked = rank_probes_by_discrimination(db, evaluator)
    t_rank_elapsed = (time.time() - t_rank_0) * 1000.0

    print("=" * 60)
    print(f"All {len(ALL_PROBES)} probe matrices precomputed in {time.time() - total_start:.2f}s!")
    print(f"Cached ranking retrieval benchmark: {t_rank_elapsed:.2f} ms")
    print("Top 3 probes by MDI:")
    for r in ranked[:3]:
        print(f"  #{r['rank']} {r['probe_id']}: MDI={r['mdi']} (Tier {r['tier']}, WinRate={r['overall_win_rate']*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
