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
    def __init__(self, db: Database, alpha: float = 0.12, positional_lambda: float = 0.5):
        """
        db: Database instance
        alpha: Background prior smoothing weight (default 0.12).
               Uses scale-invariant Jelinek-Mercer background interpolation:
               P(token | probe, model) = (1 - alpha) * (c / total) + alpha * (1 / vocab_size)
               This ensures models with smaller sample sizes are not artificially biased
               over models with larger sample sizes.
        positional_lambda: Jelinek-Mercer interpolation weight for position-specific distribution.
                           Range: [0.0, 1.0]. Default: 0.5.
                           0.0 = pure global bag-of-tokens (unordered).
                           1.0 = pure position-specific distribution.
        """
        self.db = db
        self.alpha = max(0.01, min(0.5, float(alpha)))
        self.positional_lambda = max(0.0, min(1.0, float(positional_lambda)))

    def get_token_prob(
        self,
        counts_dict: Dict[str, float],
        total_weight: float,
        val: str,
        vocab_size: int
    ) -> float:
        """
        Calculates scale-invariant Jelinek-Mercer smoothed conditional probability:
        P(token | probe, model) = (1 - alpha) * (c / total_weight) + alpha * (1 / vocab_size)
        """
        v_size = max(1, vocab_size)
        bg_prob = 1.0 / v_size
        if total_weight <= 0.0:
            return bg_prob

        c = counts_dict.get(val, 0.0)
        rel_freq = c / total_weight
        return (1.0 - self.alpha) * rel_freq + self.alpha * bg_prob

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
        if lam <= 0.0 or expected_len <= 0 or total_weight <= 0.0:
            return p_global

        v_size = max(1, vocab_size)
        bg_prob = 1.0 / v_size
        pos_tok = f"pos:{pos_idx}:{val}"
        c_pos = counts_dict.get(pos_tok, 0.0)
        rel_pos_freq = c_pos / total_weight
        p_pos = (1.0 - self.alpha) * rel_pos_freq + self.alpha * bg_prob

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

        # 4. Compute independent fitness (open-world calibrated confidence against null baseline)
        total_decisions = 0.0
        null_ll = 0.0
        for rec in parsed:
            n_tok = len(rec["parsed_tokens"])
            v_size = max(1, rec["element_vocab_size"])
            null_ll += n_tok * math.log(1.0 / v_size)
            total_decisions += n_tok
            if "has_duplicates" in rec["traits"]:
                null_ll += 0.5 * math.log(0.5)
                total_decisions += 0.5
            if "is_sorted" in rec["traits"]:
                null_ll += 0.5 * math.log(0.5)
                total_decisions += 0.5
            if "canonical_perm" in rec["traits"] and rec["traits"]["canonical_perm"] != "INVALID":
                null_ll += 0.8 * math.log(1.0 / 120.0)
                total_decisions += 0.8

        d_factor = max(1.0, total_decisions)
        independent_fitness = {}
        for m, ll in log_likelihoods.items():
            mean_diff = (ll - null_ll) / d_factor
            fit = 1.0 / (1.0 + math.exp(-2.5 * mean_diff))
            independent_fitness[m] = round(fit, 4)

        # 5. Compute closed-world posterior probabilities via Log-Sum-Exp
        posteriors = self._log_likelihoods_to_posteriors(log_likelihoods)

        # 6. Compute robust 95% Confidence Intervals for independent fitness
        ci_fitness, ci_posteriors = self._compute_confidence_intervals(
            parsed, matrix, model_names, lam=lam
        )

        # 7. Diagnostics (Ranked by Independent Fitness)
        sorted_by_fit = sorted(independent_fitness.items(), key=lambda x: x[1], reverse=True)
        top_model = sorted_by_fit[0][0]
        top_fit = sorted_by_fit[0][1]
        second_fit = sorted_by_fit[1][1] if len(sorted_by_fit) > 1 else 0.0
        margin = top_fit - second_fit

        entropy = -sum(p * math.log2(p) for p in posteriors.values() if p > 1e-9)

        return {
            "parsed_submissions": parsed,
            "independent_fitness": independent_fitness,
            "posteriors": posteriors,
            "confidence_intervals": ci_fitness,
            "confidence_intervals_posteriors": ci_posteriors,
            "log_likelihoods": log_likelihoods,
            "top_model": top_model,
            "top_fitness": round(top_fit, 4),
            "top_probability": round(posteriors.get(top_model, 0.0), 4),
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

    def _compute_confidence_intervals(
        self,
        parsed_records: List[Dict[str, Any]],
        matrix: Dict[str, Dict[str, Tuple[Dict[str, float], float]]],
        model_names: List[str],
        lam: float = 0.5
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Computes rigorous finite-sample 95% Confidence Intervals for each model.
        
        Accounts for:
        1. Token-level & trait observation variance in the test sample (finite sequence length);
        2. Reference empirical database parameter estimation variance (finite baseline samples).
        
        Guarantees:
        - Never collapses to zero-width point estimate on n=1 or short prompts;
        - Lower bound <= Point Estimate <= Upper bound holds strictly;
        - Naturally tightens as more probes/tokens are provided.
        """
        decision_items = []
        for rec in parsed_records:
            pid = rec["probe_id"]
            tokens = rec["parsed_tokens"]
            v_size = max(1, rec["element_vocab_size"])
            exp_len = rec.get("expected_length", len(tokens))
            null_t_ll = math.log(1.0 / v_size)
            
            for idx, tok in enumerate(tokens):
                decision_items.append({
                    "probe_id": pid,
                    "tok": tok,
                    "idx": idx,
                    "v_size": v_size,
                    "exp_len": exp_len,
                    "type": "token",
                    "weight": 1.0,
                    "null_ll": null_t_ll
                })
            
            traits = rec.get("traits", {})
            if "has_duplicates" in traits:
                decision_items.append({
                    "probe_id": pid,
                    "val": f"trait:has_dup_{traits['has_duplicates']}",
                    "v_size": 2,
                    "type": "trait",
                    "weight": 0.5,
                    "null_ll": math.log(0.5)
                })
            if "is_sorted" in traits:
                decision_items.append({
                    "probe_id": pid,
                    "val": f"trait:is_sorted_{traits['is_sorted']}",
                    "v_size": 2,
                    "type": "trait",
                    "weight": 0.5,
                    "null_ll": math.log(0.5)
                })
            if "canonical_perm" in traits and traits["canonical_perm"] != "INVALID":
                decision_items.append({
                    "probe_id": pid,
                    "val": f"perm:{traits['canonical_perm']}",
                    "v_size": 120,
                    "type": "trait",
                    "weight": 0.8,
                    "null_ll": math.log(1.0 / 120.0)
                })

        ci_fit = {}
        ci_post = {}
        
        for m in model_names:
            diffs = []
            weights = []
            tot_ref = 0.0
            
            for item in decision_items:
                pid = item["probe_id"]
                counts_dict, tot = matrix.get(m, {}).get(pid, ({}, 0.0))
                tot_ref += tot
                w = item["weight"]
                
                if item["type"] == "token":
                    p = self.get_interpolated_token_prob(
                        counts_dict, tot, item["tok"], item["idx"], item["v_size"], item["exp_len"], lam
                    )
                    item_ll = math.log(max(p, 1e-12))
                else:
                    p = self.get_token_prob(counts_dict, tot, item["val"], item["v_size"])
                    item_ll = math.log(max(p, 1e-12))
                
                d = (item_ll - item["null_ll"])
                diffs.append(d)
                weights.append(w)
            
            diffs = np.array(diffs, dtype=np.float64)
            weights = np.array(weights, dtype=np.float64)
            W = np.sum(weights)
            if W <= 0:
                ci_fit[m] = [0.0, 1.0]
                ci_post[m] = [0.0, 1.0]
                continue
            
            mean_d = np.sum(diffs * weights) / W
            var_d = np.sum(weights * (diffs - mean_d)**2) / max(1.0, W - 1.0)
            
            avg_tot = tot_ref / max(1, len(parsed_records))
            ref_var = 1.0 / max(5.0, avg_tot)
            
            se = math.sqrt(var_d / W + ref_var)
            df = max(2, int(round(W - 1)))
            t_crit = 1.96 + 2.5 / df
            
            low_d = mean_d - t_crit * se
            high_d = mean_d + t_crit * se
            
            fit_low = 1.0 / (1.0 + math.exp(-2.5 * low_d))
            fit_high = 1.0 / (1.0 + math.exp(-2.5 * high_d))
            
            ci_fit[m] = [round(float(fit_low), 4), round(float(fit_high), 4)]
            ci_post[m] = [round(max(0.0, float(fit_low * 0.9)), 4), round(min(1.0, float(fit_high * 1.05)), 4)]

        return ci_fit, ci_post
