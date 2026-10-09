"""
StatLLM Real Data Statistical Analysis and Cross-Validation Script.
Reads real_api_samples.json, analyzes discriminability, and runs leave-one-out cross validation.
"""

import json
from collections import defaultdict, Counter
import numpy as np

from statllm.probes import PROBES
from statllm.database import Database
from statllm.engine import LikelihoodEvaluator


def analyze():
    import os
    sample_file = "real_api_samples.json"
    if not os.path.exists(sample_file):
        sample_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real_api_samples.json")
    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {sample_file}: {e}")
        return

    print(f"Total collected real samples: {len(data)}")

    # 1. Parse all items
    by_model_probe = defaultdict(lambda: defaultdict(list))
    for item in data:
        m = item["model"]
        pid = item["probe_id"]
        raw = item["raw_text"]
        probe = PROBES.get(pid)
        if probe:
            res = probe.parse(raw)
            by_model_probe[m][pid].append({
                "raw": raw,
                "tokens": res["parsed_tokens"],
                "traits": res.get("traits", {}),
                "is_valid": res["is_valid"],
                "strictly_complied": res["strictly_complied"]
            })

    # 2. Print Characteristic Findings
    print("\n" + "=" * 60)
    print("         REAL MODEL EMPIRICAL CHARACTERISTICS")
    print("=" * 60)

    for m, probes in by_model_probe.items():
        print(f"\n--- Model: {m} ---")
        
        # Q1: arr_int5
        int_samples = probes.get("arr_int5", [])
        if int_samples:
            first_numbers = [s["traits"].get("first_token") for s in int_samples]
            all_numbers = [tok for s in int_samples for tok in s["tokens"] if tok.isdigit()]
            dup_count = sum(1 for s in int_samples if s["traits"].get("has_duplicates"))
            sorted_count = sum(1 for s in int_samples if s["traits"].get("is_sorted"))
            top_nums = Counter(all_numbers).most_common(5)

            print(f"  [Q1 整数数组 (N={len(int_samples)})]")
            print(f"    - 首元素分布: {Counter(first_numbers).most_common(3)}")
            print(f"    - 最喜爱的Top5数字: {top_nums}")
            print(f"    - 出现重复数字的比例: {dup_count}/{len(int_samples)} ({dup_count/len(int_samples)*100:.0f}%)")
            print(f"    - 严格按升序排列的比例: {sorted_count}/{len(int_samples)} ({sorted_count/len(int_samples)*100:.0f}%)")

        # Q2: arr_color5
        color_samples = probes.get("arr_color5", [])
        if color_samples:
            all_colors = [tok for s in color_samples for tok in s["tokens"]]
            top_colors = Counter(all_colors).most_common(4)
            dup_colors = sum(1 for s in color_samples if s["traits"].get("has_duplicates"))
            print(f"  [Q2 颜色数组 (N={len(color_samples)})]")
            print(f"    - Top颜色先验: {top_colors}")
            print(f"    - 出现重复颜色的比例: {dup_colors}/{len(color_samples)} ({dup_colors/len(color_samples)*100:.0f}%)")

        # Q5: arr_perm5
        perm_samples = probes.get("arr_perm5", [])
        if perm_samples:
            perms = [s["traits"].get("canonical_perm") for s in perm_samples]
            print(f"  [Q5 排列置乱 (N={len(perm_samples)})]")
            print(f"    - 置乱序列: {perms[:3]}")

    # 3. Discriminability / Likelihood Test
    print("\n" + "=" * 60)
    print("         DISCRIMINABILITY & ATTRIBUTION TEST")
    print("=" * 60)

    # Build evaluation DB on temporary SQLite
    test_db = Database("test_real.db")
    test_db.clear_database()

    # Split: use 70% as reference baseline, 30% as test queries
    ref_samples = []
    test_queries = []

    for m, probes in by_model_probe.items():
        test_db.add_model(name=m, display_name=m, provider=m)
        for pid, records in probes.items():
            n_tot = len(records)
            n_ref = max(1, int(n_tot * 0.7))
            
            for r in records[:n_ref]:
                ref_samples.append({
                    "model_name": m,
                    "probe_id": pid,
                    "raw_text": r["raw"],
                    "parsed_tokens": r["tokens"],
                    "traits": r["traits"],
                    "is_valid": r["is_valid"],
                    "strictly_complied": r["strictly_complied"],
                    "source_type": "official",
                    "weight": 1.0
                })
            for r in records[n_ref:]:
                test_queries.append({
                    "true_model": m,
                    "probe_id": pid,
                    "raw_text": r["raw"]
                })

    test_db.add_samples_batch(ref_samples)
    evaluator = LikelihoodEvaluator(test_db)

    print(f"Reference samples in test DB: {len(ref_samples)}")
    print(f"Held-out test queries: {len(test_queries)}")

    correct = 0
    total = len(test_queries)

    print("\nHeld-out Test Query Results:")
    for q in test_queries:
        res = evaluator.evaluate([{"probe_id": q["probe_id"], "raw_text": q["raw_text"]}], n_boot=200)
        pred = res["top_model"]
        prob = res["top_probability"]
        is_hit = (pred == q["true_model"])
        if is_hit:
            correct += 1
        print(f"  [{q['true_model']} on {q['probe_id']}] -> Pred: {pred} ({prob*100:.1f}%) | {'✓ HIT' if is_hit else '✗ MISS'}")

    acc = (correct / total * 100) if total > 0 else 0
    print(f"\nOverall Single-Turn Classification Accuracy: {correct}/{total} ({acc:.1f}%)")


if __name__ == "__main__":
    analyze()
