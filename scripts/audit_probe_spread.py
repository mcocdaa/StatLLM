from statllm.database import Database
from statllm.engine import LikelihoodEvaluator
from statllm.matrix import get_probe_matrix
from statllm.probes import ALL_PROBES
import numpy as np

db = Database("statllm.db")
evaluator = LikelihoodEvaluator(db)

print(f"{'Probe ID':<18} | {'Avg WinRate':<11} | {'Avg P05':<9} | {'P05 < -0.10':<12} | {'Avg P50 (Med)':<13} | {'Worst P05':<10} | {'Worst Pair'}")
print("-" * 105)

results = []

for pid in ALL_PROBES.keys():
    try:
        m_res = get_probe_matrix(db, evaluator, pid)
        matrix = m_res["matrix"]
        
        win_rates = []
        p05_list = []
        p50_list = []
        neg_p05_count = 0
        total_pairs = 0
        worst_p05 = 0.0
        worst_pair = None
        
        for mA, row in matrix.items():
            for mB, pair in row.items():
                if mA == mB:
                    continue
                total_pairs += 1
                wr = pair.get("win_rate", 0.5)
                win_rates.append(wr)
                
                quants = pair.get("quantiles", {})
                p05 = quants.get("p05", 0.0)
                p50 = quants.get("p50", 0.0)
                p05_list.append(p05)
                p50_list.append(p50)
                
                if p05 < -0.10:
                    neg_p05_count += 1
                if p05 < worst_p05:
                    worst_p05 = p05
                    worst_pair = f"{mA[:10]} vs {mB[:10]}"
                    
        avg_wr = np.mean(win_rates) if win_rates else 0.5
        avg_p05 = np.mean(p05_list) if p05_list else 0.0
        avg_p50 = np.mean(p50_list) if p50_list else 0.0
        neg_ratio = (neg_p05_count / total_pairs) * 100 if total_pairs else 0
        
        results.append({
            "pid": pid,
            "avg_wr": avg_wr,
            "avg_p05": avg_p05,
            "neg_ratio": neg_ratio,
            "avg_p50": avg_p50,
            "worst_p05": worst_p05,
            "worst_pair": worst_pair
        })
        print(f"{pid:<18} | {avg_wr*100:>8.1f}%  | {avg_p05:>8.3f}  | {neg_ratio:>9.1f}%  | {avg_p50:>13.3f} | {worst_p05:>9.3f}  | {worst_pair}")
    except Exception as e:
        print(f"{pid:<18} | ERROR: {e}")
