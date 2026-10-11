"""
StatLLM 2D Pairwise Probe Separation Matrix & Blind-Spot Diagnostic Engine.

Computes the complete 15x15 Pairwise Posterior Margin Matrix for all 9 probes:
  Delta_{i -> j} = E_{x ~ M_i} [ P(M_i | x) - P(M_j | x) ]

Analyzes:
1. 2D Margin matrix for each probe.
2. The Top-5 most confounded pairs (blind spots) for each probe.
3. Global cross-probe confusion landscape (persistent confounding pairs).
4. Mathematical insights to guide new probe design.
"""

import os
import sys
import json
import math
import sqlite3
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statllm.database import Database
from statllm.engine import LikelihoodEvaluator
from statllm.probes import PROBES


def analyze_all_probe_matrices(db_path: str = "statllm.db"):
    db = Database(db_path)
    evaluator = LikelihoodEvaluator(db, positional_lambda=0.25)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT model_name, probe_id, raw_text FROM samples WHERE is_valid = 1")
    rows = c.fetchall()
    conn.close()

    by_probe_model = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_probe_model[r["probe_id"]][r["model_name"]].append(r["raw_text"])

    model_names = sorted([m["model_name"] for m in db.get_stats()["model_breakdown"]])
    probe_ids = list(PROBES.keys())

    print("=" * 90)
    print("StatLLM 全量探针 2D 模型间两两概率优势矩阵与混淆盲区诊断报告")
    print(f"评测范围: {len(probe_ids)} 大离散探针 x {len(model_names)} 款顶尖大模型 (全量 5,544 真实样本)")
    print("=" * 90)

    # Pre-compute all evaluations by probe and model
    # probe_matrices[p_id][mA][mB] = mean_margin
    probe_matrices = {}
    probe_confounded = {}
    global_pair_confounds = defaultdict(list)

    # Load matrix once
    matrix_cache = db.get_all_model_probe_counts()

    for pid in probe_ids:
        probe = PROBES[pid]
        p_samples = by_probe_model[pid]
        margin_mat = defaultdict(lambda: defaultdict(float))
        pair_details = []

        for mA in model_names:
            s_list = p_samples.get(mA, [])
            if not s_list:
                continue

            diffs_dict = defaultdict(list)
            for raw in s_list:
                res = evaluator.evaluate([{"probe_id": pid, "raw_text": raw}], positional_lambda=0.25)
                posts = res["posteriors"]
                p_self = posts.get(mA, 0.0)
                for mB in model_names:
                    if mA != mB:
                        diffs_dict[mB].append(p_self - posts.get(mB, 0.0))

            for mB in model_names:
                if mA != mB and diffs_dict[mB]:
                    mean_d = float(np.mean(diffs_dict[mB]))
                    std_d = float(np.std(diffs_dict[mB], ddof=1)) if len(diffs_dict[mB]) > 1 else 0.0
                    n = len(diffs_dict[mB])
                    se = std_d / math.sqrt(n) if n > 0 else 0.0
                    ci = (round(mean_d - 1.96 * se, 3), round(mean_d + 1.96 * se, 3))
                    win_r = float(np.mean(np.array(diffs_dict[mB]) > 0))
                    margin_mat[mA][mB] = mean_d
                    pair_details.append({
                        "true_model": mA,
                        "comp_model": mB,
                        "margin": mean_d,
                        "ci_95": ci,
                        "win_rate": win_r
                    })

        probe_matrices[pid] = margin_mat

        # Sort pairs from most confounded (lowest margin) to best separated
        pair_details.sort(key=lambda x: x["margin"])
        probe_confounded[pid] = pair_details

        print(f"\n" + "-" * 90)
        print(f"【探针 {pid}】({probe.title}) 核心辨识力与最严重混淆对 (Top-5 盲区)")
        print("-" * 90)
        print(f"{'真实生成模型 (mA)':22} | {'被误判/胶着竞品 (mB)':22} | {'概率差均值 Δ':>12} | {'95% 置信区间':>18} | 区分状态")
        print("-" * 90)

        for p in pair_details[:5]:
            mA = p["true_model"]
            mB = p["comp_model"]
            m_val = p["margin"]
            ci_str = f"[{p['ci_95'][0]*100:+5.1f}%, {p['ci_95'][1]*100:+5.1f}%]"
            
            if m_val < 0.0:
                tag = "❌ 发生倒挂 (极易误判)"
            elif m_val < 0.05:
                tag = "⚠️ 严重混淆 (盲区)"
            elif m_val < 0.15:
                tag = "⚡ 弱区分 (相持)"
            else:
                tag = "✓ 良好"

            print(f"{mA:22} | {mB:22} | {m_val*100:+11.1f}% | {ci_str:>18} | {tag}")
            # Record global confounding if margin is low
            if m_val < 0.10:
                pair_key = tuple(sorted([mA, mB]))
                global_pair_confounds[pair_key].append((pid, m_val))

    # =========================================================================
    # PART 2: Global Cross-Probe Confusion Landscape
    # =========================================================================
    print("\n" + "=" * 90)
    print("【全局交叉透视】各探针共有混淆集群与核心痛点对 (Persistent Confounding Clusters)")
    print("=" * 90)
    print(f"{'模型冲突对 (Pair)':45} | {'混淆探针数量':>12} | 典型混淆探针与概率差")
    print("-" * 90)

    sorted_confounds = sorted(global_pair_confounds.items(), key=lambda x: len(x[1]), reverse=True)
    for (mA, mB), p_list in sorted_confounds[:10]:
        p_info = ", ".join([f"{pid}({val*100:+3.0f}%)" for pid, val in p_list[:4]])
        print(f"{mA + ' <-> ' + mB:45} | {len(p_list):12d} | {p_info}")

    print("=" * 90)


if __name__ == "__main__":
    analyze_all_probe_matrices("statllm.db")
