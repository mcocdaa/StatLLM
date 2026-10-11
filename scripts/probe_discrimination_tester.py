"""
StatLLM Probe Discrimination Tester & Pairwise Separation Benchmark.

Computes:
1. Pairwise Posterior Margin Mean & 95% CI:
   Delta_{A -> B} = P(M_A | x) - P(M_B | x) for x ~ M_A.
2. Confounded pairs detection (Margin near 0 or negative).
3. Max-Min Pairwise Separability metric for probe combinations.
"""

import os
import sys
import json
import math
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statllm.database import Database
from statllm.engine import LikelihoodEvaluator
from statllm.probes import PROBES


def compute_pairwise_probe_report(db_path: str = "statllm.db", probe_id: str = "arr_int5") -> Dict[str, Any]:
    """
    Evaluates a single probe across all model pairs in the database.
    Computes:
    - Mean margin Delta_{A -> B}
    - 95% Confidence Interval [low, high]
    - Success rate P(Delta > 0)
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT model_name, raw_text
        FROM samples
        WHERE probe_id = ? AND is_valid = 1
    """, (probe_id,))
    rows = c.fetchall()
    conn.close()

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model_name"]].append(r["raw_text"])

    models = sorted(list(by_model.keys()))
    db = Database(db_path)
    evaluator = LikelihoodEvaluator(db, positional_lambda=0.25)

    # Calculate posteriors for each model's samples
    pairwise_margins = defaultdict(lambda: defaultdict(list))
    self_posteriors = defaultdict(list)

    for m in models:
        samples = by_model[m]
        for raw in samples:
            res = evaluator.evaluate([{"probe_id": probe_id, "raw_text": raw}], positional_lambda=0.25)
            posts = res["posteriors"]
            p_self = posts.get(m, 0.0)
            self_posteriors[m].append(p_self)
            for other_m in models:
                if other_m != m:
                    p_other = posts.get(other_m, 0.0)
                    pairwise_margins[m][other_m].append(p_self - p_other)

    report = {}
    for mA in models:
        report[mA] = {}
        for mB in models:
            if mA == mB:
                continue
            diffs = np.array(pairwise_margins[mA][mB])
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            n = len(diffs)
            se = std_diff / math.sqrt(n) if n > 0 else 0.0
            ci_95 = (round(mean_diff - 1.96 * se, 4), round(mean_diff + 1.96 * se, 4))
            win_rate = float(np.mean(diffs > 0))

            report[mA][mB] = {
                "mean_margin": round(mean_diff, 4),
                "ci_95": ci_95,
                "win_rate": round(win_rate, 4),
                "sample_size": n
            }

    return {"probe_id": probe_id, "models": models, "pairwise": report}


def print_pairwise_summary(report_data: Dict[str, Any], target_model: str = "DeepSeek-V4.1-Flash"):
    pid = report_data["probe_id"]
    pairwise = report_data["pairwise"].get(target_model, {})
    if not pairwise:
        print(f"Target model {target_model} not found in report.")
        return

    print("=" * 85)
    print(f"探针 ID: {pid} | 真实输入模型: {target_model} 单题区分度优势报告")
    print("=" * 85)
    print(f"{'对比竞品模型 (Vs)':25} | {'平均概率优势 Δ':>14} | {'95% 置信区间 (CI)':>22} | {'胜率 P(Δ>0)':>11} | 区分状态")
    print("-" * 85)

    sorted_comp = sorted(pairwise.items(), key=lambda x: x[1]["mean_margin"])
    for mB, stats in sorted_comp:
        m_diff = stats["mean_margin"]
        ci = stats["ci_95"]
        wr = stats["win_rate"]
        
        if m_diff < 0.0:
            status = "❌ 发生倒挂 (易误判为竞品)"
        elif m_diff < 0.05:
            status = "⚠️ 严重混淆 (区分度低)"
        elif m_diff < 0.15:
            status = "⚡ 中等区分 (需补题)"
        else:
            status = "✅ 强力区分 (优势显著)"

        ci_str = f"[{ci[0]*100:+5.1f}%, {ci[1]*100:+5.1f}%]"
        print(f"{mB:25} | {m_diff*100:+13.1f}% | {ci_str:>22} | {wr*100:10.1f}% | {status}")
    print("=" * 85)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="arr_int5", help="Probe ID to analyze")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash", help="True model")
    args = parser.parse_args()

    rep = compute_pairwise_probe_report("statllm.db", args.probe)
    print_pairwise_summary(rep, args.model)
