"""
StatLLM Inference Engine.

Performs:
1. Parsing of raw text into discrete arrays and structural traits.
2. Dirichlet-smoothed Categorical Joint Log-Likelihood computation across array elements.
3. Multi-sample Maximum Likelihood & Posterior probability estimation.
4. Bootstrap 95% Confidence Interval (CI) calculation.
5. Statistical diagnostics (Margin, Shannon entropy, Likelihood Ratio Test).
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from statllm.probes import get_probe, PROBES
from statllm.database import Database


class LikelihoodEvaluator:
    def __init__(self, db: Database, alpha: float = 0.5, positional_lambda: float = 0.5):
        """
        db: Database instance
        alpha: Dirichlet / Jeffreys prior pseudo-count (default 0.5)
        positional_lambda: Jelinek-Mercer interpolation weight for position-specific distribution.
                           Range: [0.0, 1.0]. Default: 0.5.
                           0.0 = pure global bag-of-tokens (unordered).
                           1.0 = pure position-specific distribution.
        """
        self.db = db
        self.alpha = alpha
        self.positional_lambda = max(0.0, min(1.0, float(positional_lambda)))

    def get_token_prob(
        self,
        counts_dict: Dict[str, float],
        total_weight: float,
        val: str,
        vocab_size: int
    ) -> float:
        """
        Calculates smoothed conditional probability:
        P(token | probe, model) = (Count(token) + alpha) / (Total + alpha * vocab_size)
        """
        c = counts_dict.get(val, 0.0)
        return (c + self.alpha) / (total_weight + self.alpha * vocab_size)

    def get_interpolated_token_prob(
        self,
        counts_dict: Dict[str, float],
        total_weight: float,
        val: str,
        pos_idx: int,
        vocab_size: int,
        expected_len: int,
        lam: float
    ) -> float:
        """
        Calculates Jelinek-Mercer interpolated probability:
        P(token | pos, probe, model) = (1 - lam) * P_global(token) + lam * P_pos(token | pos)
        """
        p_global = self.get_token_prob(counts_dict, total_weight, val, vocab_size)
        if lam <= 0.0 or expected_len <= 0:
            return p_global

        pos_tok = f"pos:{pos_idx}:{val}"
        c_pos = counts_dict.get(pos_tok, 0.0)
        pos_tot = total_weight / max(1.0, float(expected_len))
        p_pos = (c_pos + self.alpha) / (pos_tot + self.alpha * vocab_size)

        if lam >= 1.0:
            return p_pos
        return (1.0 - lam) * p_global + lam * p_pos

    def evaluate(
        self,
        submissions: List[Dict[str, str]],
        positional_lambda: Optional[float] = None,
        n_boot: int = 800
    ) -> Dict[str, Any]:
        """
        Evaluates a set of user submissions containing array answers.
        """
        if not submissions:
            raise ValueError("No submissions provided for evaluation.")

        lam = self.positional_lambda if positional_lambda is None else positional_lambda
        lam = max(0.0, min(1.0, float(lam)))

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
                "parsed_tokens": parse_res["parsed_tokens"],
                "is_valid": parse_res["is_valid"],
                "strictly_complied": parse_res["strictly_complied"],
                "traits": parse_res.get("traits", {}),
                "element_vocab_size": probe.element_vocab_size,
                "expected_length": probe.expected_length
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
                tokens = record["parsed_tokens"]
                v_size = record["element_vocab_size"]
                exp_len = record["expected_length"]
                traits = record["traits"]
                
                counts_dict, tot = matrix.get(m, {}).get(pid, ({}, 0.0))
                
                # Element-wise log likelihoods (interpolated with position)
                for idx, tok in enumerate(tokens):
                    p = self.get_interpolated_token_prob(
                        counts_dict, tot, tok, idx, v_size, exp_len, lam
                    )
                    ll += math.log(max(p, 1e-12))

                # Trait log likelihoods (e.g. duplicate avoidance, sorting habits)
                if "has_duplicates" in traits:
                    dup_tok = f"trait:has_dup_{traits['has_duplicates']}"
                    p_dup = self.get_token_prob(counts_dict, tot, dup_tok, 2)
                    ll += 0.5 * math.log(max(p_dup, 1e-12))
                if "is_sorted" in traits:
                    sort_tok = f"trait:is_sorted_{traits['is_sorted']}"
                    p_sort = self.get_token_prob(counts_dict, tot, sort_tok, 2)
                    ll += 0.5 * math.log(max(p_sort, 1e-12))
                if "canonical_perm" in traits and traits["canonical_perm"] != "INVALID":
                    perm_tok = f"perm:{traits['canonical_perm']}"
                    p_perm = self.get_token_prob(counts_dict, tot, perm_tok, 120)
                    ll += 0.8 * math.log(max(p_perm, 1e-12))

            log_likelihoods[m] = ll

        # 4. Compute posterior probabilities via Log-Sum-Exp
        posteriors = self._log_likelihoods_to_posteriors(log_likelihoods)

        # 5. Bootstrap 95% Confidence Intervals
        ci = self._compute_bootstrap_ci(parsed, matrix, model_names, lam=lam, n_boot=n_boot)

        # 6. Diagnostics
        sorted_models = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        top_model = sorted_models[0][0]
        top_prob = sorted_models[0][1]
        second_prob = sorted_models[1][1] if len(sorted_models) > 1 else 0.0
        margin = top_prob - second_prob

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
            "positional_lambda": round(lam, 3),
            "sample_count": len(parsed),
            "unique_probes_tested": len(set(r["probe_id"] for r in parsed))
        }

    def _log_likelihoods_to_posteriors(self, log_likelihoods: Dict[str, float]) -> Dict[str, float]:
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
        lam: float = 0.5,
        n_boot: int = 800
    ) -> Dict[str, List[float]]:
        n = len(parsed_records)
        boot_posteriors = defaultdict(list)

        sample_lls = np.zeros((n, len(model_names)), dtype=np.float64)
        for i, record in enumerate(parsed_records):
            pid = record["probe_id"]
            tokens = record["parsed_tokens"]
            v_size = record["element_vocab_size"]
            exp_len = record.get("expected_length", len(tokens))
            traits = record["traits"]
            for j, m in enumerate(model_names):
                counts_dict, tot = matrix.get(m, {}).get(pid, ({}, 0.0))
                ll_item = 0.0
                for idx, tok in enumerate(tokens):
                    p = self.get_interpolated_token_prob(
                        counts_dict, tot, tok, idx, v_size, exp_len, lam
                    )
                    ll_item += math.log(max(p, 1e-12))
                if "has_duplicates" in traits:
                    p_dup = self.get_token_prob(counts_dict, tot, f"trait:has_dup_{traits['has_duplicates']}", 2)
                    ll_item += 0.5 * math.log(max(p_dup, 1e-12))
                if "is_sorted" in traits:
                    p_sort = self.get_token_prob(counts_dict, tot, f"trait:is_sorted_{traits['is_sorted']}", 2)
                    ll_item += 0.5 * math.log(max(p_sort, 1e-12))
                if "canonical_perm" in traits and traits["canonical_perm"] != "INVALID":
                    p_perm = self.get_token_prob(counts_dict, tot, f"perm:{traits['canonical_perm']}", 120)
                    ll_item += 0.8 * math.log(max(p_perm, 1e-12))
                sample_lls[i, j] = ll_item

        rng = np.random.default_rng(42)
        indices = rng.integers(0, n, size=(n_boot, n))
        
        for b in range(n_boot):
            boot_ll = sample_lls[indices[b]].sum(axis=0)
            max_ll = np.max(boot_ll)
            exp_ll = np.exp(boot_ll - max_ll)
            boot_p = exp_ll / np.sum(exp_ll)
            
            for j, m in enumerate(model_names):
                boot_posteriors[m].append(boot_p[j])

        ci = {}
        for m in model_names:
            arr = boot_posteriors[m]
            lower = float(np.percentile(arr, 2.5))
            upper = float(np.percentile(arr, 97.5))
            ci[m] = [round(lower, 4), round(upper, 4)]
        return ci
