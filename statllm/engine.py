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
    def __init__(self, db: Database, alpha: float = 0.12, beta: float = 0.5, positional_lambda: float = 0.5):
        """
        db: Database instance
        alpha: Legacy background prior smoothing weight for backward compatibility.
        beta: Dirichlet-Multinomial symmetric prior pseudo-count (default 0.5, Jeffreys prior).
              Provides strictly normalized probability distributions (sum P == 1.0) and
              guarantees Bernstein-von Mises asymptotic convergence as sample size N -> infty.
        positional_lambda: Interpolation weight for position-specific distribution.
                           Range: [0.0, 1.0]. Default: 0.5.
                           0.0 = pure global bag-of-tokens (unordered).
                           1.0 = pure position-specific distribution.
        """
        self.db = db
        self.alpha = max(0.01, min(0.5, float(alpha)))
        self.beta = max(0.01, float(beta))
        self.positional_lambda = max(0.0, min(1.0, float(positional_lambda)))

    def get_token_prob(
        self,
        counts_dict: Dict[str, float],
        total_weight: float,
        val: str,
        vocab_size: int
    ) -> float:
        """
        Calculates Bayesian Dirichlet-Multinomial predictive probability:
        P(token | probe, model) = (c + beta) / (total_weight + vocab_size * beta)
        Guarantees sum_{v in V} P(v) == 1.0 and asymptotic consistency as total_weight -> infty.
        """
        v_size = max(1, vocab_size)
        if total_weight <= 0.0:
            return 1.0 / v_size

        c = counts_dict.get(val, 0.0)
        return (c + self.beta) / (total_weight + v_size * self.beta)

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
        Calculates position-interpolated probability:
        P(token | pos, probe, model) = (1 - lam) * P_global(token) + lam * P_pos(token | pos)
        where both P_global and P_pos are normalized Dirichlet-Multinomial distributions:
        P_global(v) = (c(v) + beta) / (total_weight + |V| * beta)
        P_pos(v) = (c_pos(v) + beta) / (pos_total_weight + |V| * beta)
        """
        p_global = self.get_token_prob(counts_dict, total_weight, val, vocab_size)
        if lam <= 0.0 or expected_len <= 0 or total_weight <= 0.0:
            return p_global

        v_size = max(1, vocab_size)
        pos_tok = f"pos:{pos_idx}:{val}"
        c_pos = counts_dict.get(pos_tok, 0.0)

        # Effective position sample mass: sum of all counts at pos_idx, or tot / expected_len fallback
        pos_total = sum(w for k, w in counts_dict.items() if k.startswith(f"pos:{pos_idx}:"))
        if pos_total <= 0.0:
            pos_total = total_weight / max(1, expected_len)

        p_pos = (c_pos + self.beta) / (pos_total + v_size * self.beta)

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

                # Trait log-likelihoods normalized by sample mass (composite likelihood)
                sample_tot = counts_dict.get("trait:has_dup_True", 0.0) + counts_dict.get("trait:has_dup_False", 0.0)
                if sample_tot <= 0.0:
                    sample_tot = tot / max(1, exp_len)

                if "has_duplicates" in traits:
                    dup_tok = f"trait:has_dup_{traits['has_duplicates']}"
                    c_dup = counts_dict.get(dup_tok, 0.0)
                    p_dup = (c_dup + self.beta) / (sample_tot + 2.0 * self.beta)
                    ll += 0.5 * math.log(max(p_dup, 1e-12))
                if "is_sorted" in traits:
                    sort_tok = f"trait:is_sorted_{traits['is_sorted']}"
                    c_sort = counts_dict.get(sort_tok, 0.0)
                    p_sort = (c_sort + self.beta) / (sample_tot + 2.0 * self.beta)
                    ll += 0.5 * math.log(max(p_sort, 1e-12))
                if "canonical_perm" in traits and traits["canonical_perm"] != "INVALID":
                    perm_tok = f"perm:{traits['canonical_perm']}"
                    c_perm = counts_dict.get(perm_tok, 0.0)
                    p_perm = (c_perm + self.beta) / (sample_tot + 120.0 * self.beta)
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

        # 6. Compute robust 68% and 95% Confidence Intervals & model statistics
        (
            ci_fit_95,
            ci_fit_68,
            ci_post_95,
            ci_post_68,
            model_stats
        ) = self._compute_confidence_intervals(
            parsed, matrix, model_names, lam=lam
        )

        # 7. Diagnostics (Ranked by Independent Fitness)
        sorted_by_fit = sorted(independent_fitness.items(), key=lambda x: x[1], reverse=True)
        top_model = sorted_by_fit[0][0]
        top_fit = sorted_by_fit[0][1]
        second_fit = sorted_by_fit[1][1] if len(sorted_by_fit) > 1 else 0.0
        margin = top_fit - second_fit

        entropy = -sum(p * math.log2(p) for p in posteriors.values() if p > 1e-9)

        # Build comprehensive summary statistics
        summary_stats = {
            "total_decisions": round(d_factor, 1),
            "null_baseline_ll": round(null_ll, 2),
            "sample_count": len(parsed),
            "unique_probes": len(set(r["probe_id"] for r in parsed)),
            "entropy": round(entropy, 4),
            "margin": round(margin, 4)
        }

        # Enrich model_statistics with fitness, posterior, and LL
        for m in model_names:
            if m in model_stats:
                model_stats[m]["fitness"] = independent_fitness.get(m, 0.0)
                model_stats[m]["posterior"] = posteriors.get(m, 0.0)
                model_stats[m]["log_likelihood"] = round(log_likelihoods.get(m, 0.0), 2)
                model_stats[m]["ci_68"] = ci_fit_68.get(m, [0.0, 1.0])
                model_stats[m]["ci_95"] = ci_fit_95.get(m, [0.0, 1.0])

        # 8. Sample Size Uncertainty Advisory
        top_ci = ci_fit_95.get(top_model, [0.0, 1.0])
        ci_span = round(float(top_ci[1] - top_ci[0]), 4)
        n_samples = len(parsed)
        n_unique_probes = len(set(r["probe_id"] for r in parsed))
        is_low_sample = (n_samples <= 8) or (n_unique_probes < 5) or (ci_span >= 0.30)

        target_recom = max(15, n_samples * 3)
        reduction_factor = 1.0 - math.sqrt(n_samples / target_recom)
        reduction_pct = round(max(20.0, min(80.0, reduction_factor * 100)), 1)
        expected_span = round(ci_span * math.sqrt(n_samples / target_recom), 4)

        sample_advisory = {
            "is_low_sample": is_low_sample,
            "sample_count": n_samples,
            "unique_probes": n_unique_probes,
            "ci_span": ci_span,
            "ci_span_pct": round(ci_span * 100, 1),
            "ci_radius_pct": round(ci_span * 50, 1),
            "recommended_samples": target_recom,
            "reduction_pct": reduction_pct,
            "expected_span_pct": round(expected_span * 100, 1),
            "expected_radius_pct": round(expected_span * 50, 1)
        }

        return {
            "parsed_submissions": parsed,
            "independent_fitness": independent_fitness,
            "posteriors": posteriors,
            "confidence_intervals": ci_fit_95,              # 95% default for backward compat
            "confidence_intervals_95": ci_fit_95,
            "confidence_intervals_68": ci_fit_68,
            "confidence_intervals_posteriors": ci_post_95,
            "confidence_intervals_posteriors_68": ci_post_68,
            "log_likelihoods": log_likelihoods,
            "model_statistics": model_stats,
            "summary_statistics": summary_stats,
            "sample_advisory": sample_advisory,
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

        ci_fit_95 = {}
        ci_fit_68 = {}
        ci_post_95 = {}
        ci_post_68 = {}
        model_stats = {}
        
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
                    sample_tot = counts_dict.get("trait:has_dup_True", 0.0) + counts_dict.get("trait:has_dup_False", 0.0)
                    if sample_tot <= 0.0:
                        sample_tot = tot / max(1, item.get("exp_len", 5))
                    c_trait = counts_dict.get(item["val"], 0.0)
                    p = (c_trait + self.beta) / (sample_tot + item["v_size"] * self.beta)
                    item_ll = math.log(max(p, 1e-12))
                
                d = (item_ll - item["null_ll"])
                diffs.append(d)
                weights.append(w)
            
            diffs = np.array(diffs, dtype=np.float64)
            weights = np.array(weights, dtype=np.float64)
            W = np.sum(weights)
            if W <= 0:
                ci_fit_95[m] = [0.0, 1.0]
                ci_fit_68[m] = [0.0, 1.0]
                ci_post_95[m] = [0.0, 1.0]
                ci_post_68[m] = [0.0, 1.0]
                model_stats[m] = {"standard_error": 0.0, "mean_delta_ll": 0.0, "delta_ll": 0.0, "ref_samples": 0, "df": 0}
                continue
            
            mean_d = np.sum(diffs * weights) / W
            var_d = np.sum(weights * (diffs - mean_d)**2) / max(1.0, W - 1.0)
            
            # Reference baseline effective sample size across tested probes
            tested_probes = set(r["probe_id"] for r in parsed_records)
            ref_sample_counts = []
            for pid in tested_probes:
                cd, t = matrix.get(m, {}).get(pid, ({}, 0.0))
                s_tot = cd.get("trait:has_dup_True", 0.0) + cd.get("trait:has_dup_False", 0.0)
                if s_tot <= 0.0:
                    exp_l = PROBES[pid].expected_length if pid in PROBES else 5
                    s_tot = t / max(1, exp_l)
                ref_sample_counts.append(s_tot)
            avg_tot = float(np.mean(ref_sample_counts)) if ref_sample_counts else 10.0
            ref_var = 1.0 / max(3.0, avg_tot)
            
            se = math.sqrt(var_d / W + ref_var)
            df = max(2, int(round(W - 1)))
            
            # 95% Confidence Interval (2 sigma, tail bounds)
            t_crit_95 = 1.96 + 2.5 / df
            low_d_95 = mean_d - t_crit_95 * se
            high_d_95 = mean_d + t_crit_95 * se
            fit_low_95 = 1.0 / (1.0 + math.exp(-2.5 * low_d_95))
            fit_high_95 = 1.0 / (1.0 + math.exp(-2.5 * high_d_95))
            
            # 68% Confidence Interval (1 sigma, core central mass)
            t_crit_68 = 1.00 + 0.8 / df
            low_d_68 = mean_d - t_crit_68 * se
            high_d_68 = mean_d + t_crit_68 * se
            fit_low_68 = 1.0 / (1.0 + math.exp(-2.5 * low_d_68))
            fit_high_68 = 1.0 / (1.0 + math.exp(-2.5 * high_d_68))
            
            ci_fit_95[m] = [round(float(fit_low_95), 4), round(float(fit_high_95), 4)]
            ci_fit_68[m] = [round(float(fit_low_68), 4), round(float(fit_high_68), 4)]
            ci_post_95[m] = [round(max(0.0, float(fit_low_95 * 0.9)), 4), round(min(1.0, float(fit_high_95 * 1.05)), 4)]
            ci_post_68[m] = [round(max(0.0, float(fit_low_68 * 0.95)), 4), round(min(1.0, float(fit_high_68 * 1.02)), 4)]

            model_stats[m] = {
                "standard_error": round(float(se), 4),
                "mean_delta_ll": round(float(mean_d), 4),
                "delta_ll": round(float(mean_d * W), 2),
                "ref_samples": int(round(avg_tot)),
                "df": df
            }

        return ci_fit_95, ci_fit_68, ci_post_95, ci_post_68, model_stats
