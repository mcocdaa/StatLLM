"""
StatLLM Inference Engine.

Performs:
1. Regex extraction on user submissions.
2. Dirichlet-smoothed Categorical Log-Likelihood computation.
3. Multi-sample joint Maximum Likelihood / Posterior probability estimation.
4. Bootstrap 95% Confidence Interval (CI) calculation.
5. Statistical diagnostics (Margin of victory, Shannon entropy, Likelihood Ratio Test).
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from statllm.probes import get_probe, PROBES
from statllm.database import Database


class LikelihoodEvaluator:
    def __init__(self, db: Database, alpha: float = 0.5):
        """
        db: Database instance
        alpha: Dirichlet / Jeffreys prior pseudo-count (default 0.5)
        """
        self.db = db
        self.alpha = alpha

    def get_prob(
        self,
        counts_dict: Dict[str, float],
        total_weight: float,
        val: str,
        vocab_size: int
    ) -> float:
        """
        Calculates smoothed conditional probability:
        P(val | probe, model) = (Count(val) + alpha) / (Total + alpha * vocab_size)
        """
        c = counts_dict.get(val, 0.0)
        return (c + self.alpha) / (total_weight + self.alpha * vocab_size)

    def evaluate(
        self,
        submissions: List[Dict[str, str]],
        n_boot: int = 1000
    ) -> Dict[str, Any]:
        """
        Evaluates a set of user submissions.
        
        submissions: list of dicts with {"probe_id": str, "raw_text": str}
        n_boot: number of bootstrap iterations for 95% confidence intervals
        
        Returns:
            parsed_submissions: list of parsed records
            posteriors: Dict[model_name, float]
            confidence_intervals: Dict[model_name, [lower_ci, upper_ci]]
            log_likelihoods: Dict[model_name, float]
            top_model: str
            confidence_score: float (percentage margin)
            entropy: float
        """
        if not submissions:
            raise ValueError("No submissions provided for evaluation.")

        # 1. Parse all user submissions
        parsed = []
        for item in submissions:
            pid = item.get("probe_id")
            raw = item.get("raw_text", "")
            probe = get_probe(pid)
            if not probe:
                raise ValueError(f"Unknown probe ID: {pid}")
            
            parse_res = probe.parse(raw)
            parsed.append({
                "probe_id": pid,
                "raw_text": raw,
                "parsed_value": parse_res["parsed_value"],
                "is_valid": parse_res["is_valid"],
                "strictly_complied": parse_res["strictly_complied"],
                "vocab_size": probe.vocab_size
            })

        # 2. Load count table from database
        matrix = self.db.get_all_model_probe_counts()
        models = self.db.list_models()
        model_names = [m["name"] for m in models] if models else list(matrix.keys())

        if not model_names:
            raise ValueError("No baseline models found in database. Please seed the database first.")

        # 3. Compute joint log-likelihood for each model
        log_likelihoods = {}
        for m in model_names:
            ll = 0.0
            for record in parsed:
                pid = record["probe_id"]
                val = record["parsed_value"]
                v_size = record["vocab_size"]
                
                counts_dict, tot = matrix.get(m, {}).get(pid, ({}, 0.0))
                p = self.get_prob(counts_dict, tot, val, v_size)
                ll += math.log(max(p, 1e-12))
            log_likelihoods[m] = ll

        # 4. Compute posterior probabilities via Log-Sum-Exp
        posteriors = self._log_likelihoods_to_posteriors(log_likelihoods)

        # 5. Bootstrap 95% Confidence Intervals
        ci = self._compute_bootstrap_ci(parsed, matrix, model_names, n_boot=n_boot)

        # 6. Diagnostics: Margin, Shannon Entropy, Top Model
        sorted_models = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        top_model = sorted_models[0][0]
        top_prob = sorted_models[0][1]
        second_prob = sorted_models[1][1] if len(sorted_models) > 1 else 0.0
        margin = top_prob - second_prob

        # Shannon Entropy H(P) = -sum(p * log2(p))
        entropy = -sum(p * math.log2(p) for p in posteriors.values() if p > 1e-9)

        return {
            "parsed_submissions": parsed,
            "posteriors": posteriors,
            "confidence_intervals": ci,
            "log_likelihoods": log_likelihoods,
            "top_model": top_model,
            "top_probability": round(top_prob, 4),
            "margin": round(margin, 4),
            "entropy": round(entropy, 4),
            "sample_count": len(parsed),
            "unique_probes_tested": len(set(r["probe_id"] for r in parsed))
        }

    def _log_likelihoods_to_posteriors(self, log_likelihoods: Dict[str, float]) -> Dict[str, float]:
        """Converts log likelihoods to normalized posterior probabilities using Log-Sum-Exp."""
        if not log_likelihoods:
            return {}
        max_ll = max(log_likelihoods.values())
        exps = {m: math.exp(ll - max_ll) for m, ll in log_likelihoods.items()}
        sum_exp = sum(exps.values())
        return {m: exps[m] / sum_exp for m in log_likelihoods}

    def _compute_bootstrap_ci(
        self,
        parsed_records: List[Dict[str, Any]],
        matrix: Dict[str, Dict[str, Tuple[Dict[str, float], float]]],
        model_names: List[str],
        n_boot: int = 1000
    ) -> Dict[str, List[float]]:
        """
        Computes 95% empirical bootstrap confidence intervals [2.5%, 97.5%].
        If n=1, interval reflects the uncertainty of a single observation.
        """
        n = len(parsed_records)
        boot_posteriors = defaultdict(list)

        # Precompute per-sample log-likelihood array: shape (n_samples, n_models)
        sample_lls = np.zeros((n, len(model_names)), dtype=np.float64)
        for i, record in enumerate(parsed_records):
            pid = record["probe_id"]
            val = record["parsed_value"]
            v_size = record["vocab_size"]
            for j, m in enumerate(model_names):
                counts_dict, tot = matrix.get(m, {}).get(pid, ({}, 0.0))
                p = self.get_prob(counts_dict, tot, val, v_size)
                sample_lls[i, j] = math.log(max(p, 1e-12))

        # Bootstrap resampling
        rng = np.random.default_rng(42)
        indices = rng.integers(0, n, size=(n_boot, n))
        
        for b in range(n_boot):
            # Sum log likelihoods of resampled indices
            boot_ll = sample_lls[indices[b]].sum(axis=0)
            max_ll = np.max(boot_ll)
            exp_ll = np.exp(boot_ll - max_ll)
            boot_p = exp_ll / np.sum(exp_ll)
            
            for j, m in enumerate(model_names):
                boot_posteriors[m].append(boot_p[j])

        # Extract 2.5% and 97.5% quantiles
        ci = {}
        for m in model_names:
            arr = boot_posteriors[m]
            lower = float(np.percentile(arr, 2.5))
            upper = float(np.percentile(arr, 97.5))
            ci[m] = [round(lower, 4), round(upper, 4)]
        return ci
