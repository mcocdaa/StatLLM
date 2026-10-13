"""
StatLLM 2D Pairwise Margin Matrix & Single-Probe Discrimination Diagnostics.
"""

import math
import sqlite3
from collections import defaultdict
from typing import Dict, Any, List, Optional
import numpy as np

from statllm.database import Database
from statllm.engine import LikelihoodEvaluator
from statllm.probes import get_probe

_MATRIX_CACHE: Dict[str, Dict[str, Any]] = {}
_RANKING_CACHE: Optional[List[Dict[str, Any]]] = None


def get_probe_matrix(
    db: Database,
    evaluator: LikelihoodEvaluator,
    probe_id: str,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Computes or retrieves cached 2D 15x15 Pairwise Posterior Margin Matrix
    for a specific probe across all empirical baseline samples.
    """
    global _MATRIX_CACHE
    if not force_refresh and probe_id in _MATRIX_CACHE:
        return _MATRIX_CACHE[probe_id]

    if not force_refresh:
        db_cached = db.get_probe_matrix_cache(probe_id)
        if db_cached:
            _MATRIX_CACHE[probe_id] = db_cached
            return db_cached

    probe = get_probe(probe_id)
    if not probe:
        raise ValueError(f"Unknown probe ID: {probe_id}")

    with db._get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT model_name, raw_text FROM samples WHERE is_valid = 1 AND probe_id = ?",
            (probe_id,)
        )
        rows = c.fetchall()

    data = defaultdict(list)
    for r in rows:
        data[r["model_name"]].append(r["raw_text"])

    models = sorted(list(data.keys()))
    matrix: Dict[str, Dict[str, Any]] = {}
    pair_details: List[Dict[str, Any]] = []
    all_margins: List[float] = []
    all_win_rates: List[float] = []

    full_counts = db.get_all_model_probe_counts()

    for mA in models:
        s_list = data[mA]
        c_dict, tot = full_counts.get(mA, {}).get(probe_id, ({}, 0.0))
        posts_list = []
        for raw in s_list:
            parse_res = probe.parse(raw)
            tokens = parse_res.get("parsed_tokens", [])
            traits = parse_res.get("traits", {})

            # Subtract this test sample's counts for strict out-of-sample generalization evaluation
            c_oos = dict(c_dict)
            for idx, tok in enumerate(tokens):
                t_str = str(tok)
                c_oos[t_str] = max(0.0, c_oos.get(t_str, 0.0) - 1.0)
                c_oos[f"pos:{idx}:{t_str}"] = max(0.0, c_oos.get(f"pos:{idx}:{t_str}", 0.0) - 1.0)
                if probe_id == "arr_int_big5":
                    if len(t_str) >= 2:
                        c_oos[f"pref:{t_str[:2]}"] = max(0.0, c_oos.get(f"pref:{t_str[:2]}", 0.0) - 1.0)
                        c_oos[f"pos_pref:{idx}:{t_str[:2]}"] = max(0.0, c_oos.get(f"pos_pref:{idx}:{t_str[:2]}", 0.0) - 1.0)
                    if len(t_str) >= 1:
                        c_oos[f"lead:{t_str[0]}"] = max(0.0, c_oos.get(f"lead:{t_str[0]}", 0.0) - 1.0)
                elif probe_id == "arr_pseudo_cvc5" and len(t_str) == 3:
                    c_oos[f"init:{t_str[0]}"] = max(0.0, c_oos.get(f"init:{t_str[0]}", 0.0) - 1.0)
                    c_oos[f"vow:{t_str[1]}"] = max(0.0, c_oos.get(f"vow:{t_str[1]}", 0.0) - 1.0)
                    c_oos[f"coda:{t_str[2]}"] = max(0.0, c_oos.get(f"coda:{t_str[2]}", 0.0) - 1.0)

            if "has_duplicates" in traits:
                c_oos[f"trait:has_dup_{traits['has_duplicates']}"] = max(0.0, c_oos.get(f"trait:has_dup_{traits['has_duplicates']}", 0.0) - 1.0)
            if "is_sorted" in traits:
                c_oos[f"trait:is_sorted_{traits['is_sorted']}"] = max(0.0, c_oos.get(f"trait:is_sorted_{traits['is_sorted']}", 0.0) - 1.0)
            if "has_consecutive" in traits and traits["has_consecutive"] is not None:
                c_oos[f"trait:consec_{traits['has_consecutive']}"] = max(0.0, c_oos.get(f"trait:consec_{traits['has_consecutive']}", 0.0) - 1.0)

            m_oos = dict(full_counts)
            m_oos[mA] = dict(full_counts.get(mA, {}))
            m_oos[mA][probe_id] = (c_oos, max(1.0, tot - len(tokens)))

            res = evaluator.evaluate([{"probe_id": probe_id, "raw_text": raw}], matrix=m_oos, positional_lambda=0.25)
            posts_list.append(res["posteriors"])

        matrix[mA] = {}
        for mB in models:
            if mA == mB:
                continue
            diffs = [p.get(mA, 0.0) - p.get(mB, 0.0) for p in posts_list]
            d_arr = np.array(diffs)
            mean_d = float(np.mean(d_arr))
            med_d = float(np.median(d_arr))
            std_d = float(np.std(d_arr, ddof=1)) if len(d_arr) > 1 else 0.0
            n = len(d_arr)
            se = std_d / math.sqrt(n) if n > 0 else 0.0
            mean_ci = [round(mean_d - 1.96 * se, 4), round(mean_d + 1.96 * se, 4)]
            win_r = float(np.mean(d_arr > 0))

            p05 = float(np.percentile(d_arr, 5)) if n > 0 else 0.0
            p25 = float(np.percentile(d_arr, 25)) if n > 0 else 0.0
            p75 = float(np.percentile(d_arr, 75)) if n > 0 else 0.0
            p95 = float(np.percentile(d_arr, 95)) if n > 0 else 0.0

            pair_info = {
                "true_model": mA,
                "comp_model": mB,
                "mean_margin": round(mean_d, 4),
                "median_margin": round(med_d, 4),
                "quantiles": {
                    "p05": round(p05, 4),
                    "p25": round(p25, 4),
                    "p50": round(med_d, 4),
                    "p75": round(p75, 4),
                    "p95": round(p95, 4)
                },
                "p05_p95_range": [round(p05, 4), round(p95, 4)],
                "iqr_range": [round(p25, 4), round(p75, 4)],
                "mean_ci_95": mean_ci,
                "ci_95": [round(p05, 4), round(p95, 4)],
                "win_rate": round(win_r, 4),
                "samples_count": n
            }
            matrix[mA][mB] = pair_info
            pair_details.append(pair_info)
            all_margins.append(mean_d)
            all_win_rates.append(win_r)

    pair_details.sort(key=lambda x: x["mean_margin"])
    top_blind_spots = pair_details[:5] if pair_details else []
    worst_pair = [top_blind_spots[0]["true_model"], top_blind_spots[0]["comp_model"]] if top_blind_spots else []

    avg_m = float(np.mean(all_margins)) if all_margins else 0.0
    avg_w = float(np.mean(all_win_rates)) if all_win_rates else 0.0
    n_pairs = len(all_margins)
    inv_count = int(np.sum(np.array(all_margins) < 0))
    inv_rate = inv_count / n_pairs if n_pairs > 0 else 0.0

    # Matrix Discrimination Index (MDI): 100 * WinRate * max(0, MeanMargin) * (1 - InversionRate)
    mdi = round(100.0 * avg_w * max(0.0, avg_m) * (1.0 - inv_rate), 2)
    tier = "S" if mdi >= 80.0 else ("A" if mdi >= 30.0 else ("B" if mdi >= 10.0 else ("C" if mdi >= 5.0 else "D")))

    res_payload = {
        "probe_id": probe_id,
        "title": probe.title,
        "mdi": mdi,
        "tier": tier,
        "overall_avg_margin": round(avg_m, 4),
        "overall_median_margin": round(float(np.median(all_margins)), 4) if all_margins else 0.0,
        "overall_win_rate": round(avg_w, 4),
        "min_bottleneck_margin": round(float(np.min(all_margins)), 4) if all_margins else 0.0,
        "bottleneck_pair": worst_pair,
        "inversions_count": inv_count,
        "inversion_rate": round(inv_rate, 4),
        "models": models,
        "matrix": matrix,
        "top_blind_spots": top_blind_spots
    }
    db.set_probe_matrix_cache(probe_id, res_payload, mdi)
    _MATRIX_CACHE[probe_id] = res_payload
    return res_payload


def rank_probes_by_discrimination(
    db: Database,
    evaluator: LikelihoodEvaluator,
    probe_ids: Optional[List[str]] = None,
    force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """
    Computes MDI for specified or all available probes in the database,
    returning a sorted ranking list (highest MDI first).
    Loads from persistent SQLite cache in 2ms when available.
    """
    global _RANKING_CACHE
    if not force_refresh and probe_ids is None and _RANKING_CACHE is not None:
        return _RANKING_CACHE

    from statllm.probes import ALL_PROBES
    if probe_ids is None:
        with db._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT DISTINCT probe_id FROM samples WHERE is_valid = 1")
            db_pids = [r[0] for r in c.fetchall()]
            probe_ids = [pid for pid in db_pids if pid in ALL_PROBES]

    all_cached = db.get_all_probe_matrices_cache() if not force_refresh else {}
    ranked = []
    for pid in probe_ids:
        try:
            m = _MATRIX_CACHE.get(pid) or all_cached.get(pid)
            if not m or force_refresh:
                m = get_probe_matrix(db, evaluator, pid, force_refresh=force_refresh)
            ranked.append({
                "probe_id": pid,
                "title": m["title"],
                "mdi": m["mdi"],
                "tier": m["tier"],
                "overall_win_rate": m["overall_win_rate"],
                "overall_avg_margin": m["overall_avg_margin"],
                "overall_median_margin": m["overall_median_margin"],
                "min_bottleneck_margin": m["min_bottleneck_margin"],
                "bottleneck_pair": m["bottleneck_pair"],
                "inversions_count": m["inversions_count"],
            })
        except Exception:
            continue

    ranked.sort(key=lambda x: x["mdi"], reverse=True)
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank

    if probe_ids is None or len(probe_ids) == len(ALL_PROBES):
        _RANKING_CACHE = ranked

    return ranked


def warm_up_matrix_cache(db: Database):
    """Loads all precomputed probe matrices from database into memory cache."""
    global _MATRIX_CACHE
    try:
        all_cached = db.get_all_probe_matrices_cache()
        _MATRIX_CACHE.update(all_cached)
    except Exception:
        pass


def clear_matrix_cache():
    """Clears in-memory cache of pairwise matrices and rankings."""
    global _MATRIX_CACHE, _RANKING_CACHE
    _MATRIX_CACHE.clear()
    _RANKING_CACHE = None


