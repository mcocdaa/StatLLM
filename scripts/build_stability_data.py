"""
Build precomputed empirical probe downside tail stability matrix for StatLLM.
"""

import pprint
from statllm.database import Database
from statllm.engine import LikelihoodEvaluator
from statllm.matrix import get_probe_matrix
from statllm.probes import ALL_PROBES

def main():
    db = Database("statllm.db")
    evaluator = LikelihoodEvaluator(db)

    stability_cache = {}

    for pid in ALL_PROBES.keys():
        m_res = get_probe_matrix(db, evaluator, pid)
        matrix = m_res["matrix"]
        
        pair_p05 = {}
        downside_list = []
        
        for mA, row in matrix.items():
            pair_p05[mA] = {}
            for mB, pair in row.items():
                if mA == mB:
                    continue
                p05 = pair.get("quantiles", {}).get("p05", 0.0)
                pair_p05[mA][mB] = round(float(p05), 4)
                downside_list.append(max(0.0, -p05))
                
        avg_downside = sum(downside_list) / len(downside_list) if downside_list else 0.0
        global_stab = max(0.05, min(1.0, 1.0 - avg_downside))
        
        stability_cache[pid] = {
            "global_stability": round(float(global_stab), 4),
            "pairwise_p05": pair_p05
        }

    header = '"""\nStatLLM Precomputed Empirical Probe Downside Tail Stability Matrix.\nAuto-generated from out-of-sample pairwise quantile benchmarks.\n"""\n\nPROBE_STABILITY_DATA = '
    code = header + pprint.pformat(stability_cache, indent=2, width=120) + "\n"

    with open("statllm/stability_data.py", "w", encoding="utf-8") as f:
        f.write(code)

    print("Saved statllm/stability_data.py successfully!")

if __name__ == "__main__":
    main()
