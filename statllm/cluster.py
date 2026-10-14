"""
StatLLM Dimension Reduction & Clustering Visualizer for Array Probes.
"""

from collections import defaultdict
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA

from statllm.database import Database
from statllm.probes import PROBES


class ClusterProjector:
    def __init__(self, db: Database):
        self.db = db
        self.pca: Optional[PCA] = None
        self.feature_keys: List[Tuple[str, str]] = []
        self.probe_slices: Dict[str, Tuple[int, int]] = {}
        self.model_centers_2d: Dict[str, List[float]] = {}
        self.model_clouds_2d: Dict[str, List[List[float]]] = {}
        self.model_base_vecs: Dict[str, np.ndarray] = {}
        self.is_fitted = False

    def _build_feature_vocabulary(self) -> None:
        self.feature_keys = []
        self.probe_slices = {}
        cur_idx = 0
        for pid, probe in PROBES.items():
            st = cur_idx
            if probe.allowed_elements:
                for val in probe.allowed_elements:
                    self.feature_keys.append((pid, val))
                    cur_idx += 1
            elif pid == "arr_pseudo_cvc5":
                for c in "BCDFGHJKLMNPQRSTVWXYZ":
                    self.feature_keys.append((pid, f"init:{c}"))
                    cur_idx += 1
                for v in "AEIOU":
                    self.feature_keys.append((pid, f"vow:{v}"))
                    cur_idx += 1
                for c in "BCDFGHJKLMNPQRSTVWXYZ":
                    self.feature_keys.append((pid, f"coda:{c}"))
                    cur_idx += 1
            elif pid == "arr_int_big5":
                for d in range(1, 10):
                    self.feature_keys.append((pid, f"lead:{d}"))
                    cur_idx += 1
                for p in range(10, 100):
                    self.feature_keys.append((pid, f"pref:{p}"))
                    cur_idx += 1
            # Include behavioral trait markers
            self.feature_keys.append((pid, "trait:has_dup_True"))
            cur_idx += 1
            self.feature_keys.append((pid, "trait:has_dup_False"))
            cur_idx += 1
            self.feature_keys.append((pid, "INVALID"))
            cur_idx += 1
            self.probe_slices[pid] = (st, cur_idx)

    @staticmethod
    def _accumulate_sample_tokens(pid: str, toks: List[Any], cnt: Dict[str, float]) -> None:
        for tok in toks:
            tok_str = str(tok)
            cnt[tok_str] += 1.0
            if pid == "arr_pseudo_cvc5" and len(tok_str) == 3:
                cnt[f"init:{tok_str[0]}"] += 1.0
                cnt[f"vow:{tok_str[1]}"] += 1.0
                cnt[f"coda:{tok_str[2]}"] += 1.0
            elif pid == "arr_int_big5":
                if len(tok_str) >= 1:
                    cnt[f"lead:{tok_str[0]}"] += 1.0
                if len(tok_str) >= 2:
                    cnt[f"pref:{tok_str[:2]}"] += 1.0

    def _vectorize_probe_profile(
        self,
        counts_by_probe: Dict[str, Tuple[Dict[str, float], float]],
        adaptive_smoothing: bool = False
    ) -> np.ndarray:
        """
        Builds a normalized multi-probe fingerprint vector using Hellinger / square-root embedding.
        Under Chentsov's Theorem, mapping probability vectors to the Hellinger sphere (v_i = sqrt(p_i))
        isometrically embeds the Fisher information metric into Euclidean space.
        Each probe's distribution is normalized on the unit sphere, ensuring balanced weighting.
        """
        d = len(self.feature_keys)
        vec = np.zeros(d, dtype=np.float64)
        for pid, (st, ed) in self.probe_slices.items():
            sub_vec = np.zeros(ed - st, dtype=np.float64)
            c_dict, tot = counts_by_probe.get(pid, ({}, 0.0))
            v_size = max(1, ed - st)
            alpha = (0.5 * min(1.0, max(0.04, tot / 40.0))) if adaptive_smoothing else 0.5
            for i, idx in enumerate(range(st, ed)):
                key = self.feature_keys[idx]
                val = key[1]
                c = c_dict.get(val, 0.0)
                prob = (c + alpha) / (tot + alpha * v_size)
                sub_vec[i] = math.sqrt(max(1e-12, prob))
            sub_norm = np.linalg.norm(sub_vec)
            if sub_norm > 0:
                vec[st:ed] = sub_vec / sub_norm
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def fit(self, n_points_per_model: Optional[int] = None):
        self._build_feature_vocabulary()
        matrix = self.db.get_all_model_probe_counts()
        models = self.db.list_models()
        model_names = [m["name"] for m in models] if models else list(matrix.keys())

        if len(model_names) < 2:
            return

        all_samples = self.db.get_all_samples()
        pts_count = n_points_per_model if n_points_per_model is not None else 35

        # Check if we have empirical samples to bootstrap from
        model_probe_samples = defaultdict(lambda: defaultdict(list))
        for s in all_samples:
            if s.get("is_valid", 1):
                model_probe_samples[s["model_name"]][s["probe_id"]].append(s)

        has_empirical_samples = any(len(p_map) > 0 for p_map in model_probe_samples.values())

        rng = np.random.default_rng(2026)
        all_fit_vecs = []
        self.model_base_vecs = {}

        if has_empirical_samples:
            for m in model_names:
                base_vec = self._vectorize_probe_profile(matrix.get(m, {}))
                self.model_base_vecs[m] = base_vec
                all_fit_vecs.append(base_vec)

                p_samples_dict = model_probe_samples[m]
                for _ in range(pts_count):
                    b_cnts = {}
                    for pid in self.probe_slices.keys():
                        s_list = p_samples_dict.get(pid, [])
                        if not s_list:
                            continue
                        resamp = rng.choice(s_list, size=len(s_list), replace=True)
                        cnt = defaultdict(float)
                        tok_tot = 0.0
                        for s in resamp:
                            try:
                                toks = json.loads(s["parsed_value"])
                            except Exception:
                                toks = []
                            self._accumulate_sample_tokens(pid, toks, cnt)
                            tok_tot += len(toks)
                            has_dup = len(toks) != len(set(toks))
                            cnt[f"trait:has_dup_{has_dup}"] += 0.5
                        b_cnts[pid] = (cnt, tok_tot)
                    all_fit_vecs.append(self._vectorize_probe_profile(b_cnts))
        else:
            # Synthetic Dirichlet jitter fallback for minimal test fixtures
            for m in model_names:
                base_vec = self._vectorize_probe_profile(matrix.get(m, {}))
                self.model_base_vecs[m] = base_vec
                all_fit_vecs.append(base_vec)
                d = len(self.feature_keys)
                for _ in range(pts_count):
                    noise = rng.normal(0, 0.03, size=d)
                    sample_vec = np.clip(base_vec + noise, 0.001, 1.0)
                    norm = np.linalg.norm(sample_vec)
                    all_fit_vecs.append(sample_vec / norm if norm > 0 else sample_vec)

        all_fit_vecs = np.array(all_fit_vecs)
        self.pca = PCA(n_components=2, random_state=2026)
        self.pca.fit(all_fit_vecs)

        # Compute distinct cluster centroids and 2D scatter clouds
        self.model_centers_2d = {}
        self.model_clouds_2d = {}

        block_size = pts_count + 1
        for i, m in enumerate(model_names):
            block = self.pca.transform(all_fit_vecs[i * block_size : (i + 1) * block_size])
            center = block[0]  # Base empirical model profile
            cloud = block[1:]  # Variation cloud
            self.model_centers_2d[m] = [round(float(center[0]), 3), round(float(center[1]), 3)]
            self.model_clouds_2d[m] = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in cloud]

        self.is_fitted = True
        self._cached_base_clusters = self._build_clusters_payload()

    def _build_clusters_payload(self) -> List[Dict[str, Any]]:
        models_meta = {m["name"]: m for m in self.db.list_models()}
        clusters = []
        for m_name, points in self.model_clouds_2d.items():
            meta = models_meta.get(m_name, {})
            clusters.append({
                "model_name": m_name,
                "display_name": meta.get("display_name", m_name),
                "color": meta.get("color", "#3b82f6"),
                "center": self.model_centers_2d.get(m_name, [0.0, 0.0]),
                "sample_count": self.db.get_sample_count(m_name),
                "points": points
            })
        return clusters

    def project_user_submission(
        self,
        parsed_submissions: List[Dict[str, Any]],
        posteriors: Optional[Dict[str, float]] = None
    ) -> Optional[List[float]]:
        if not self.is_fitted or self.pca is None or not self.feature_keys:
            self.fit()
            if not self.is_fitted or self.pca is None:
                return None

        user_counts = defaultdict(lambda: (defaultdict(float), 0.0))
        for s in parsed_submissions:
            pid = s.get("probe_id")
            if not pid or pid not in self.probe_slices:
                continue
            cnt, tot = user_counts[pid]
            toks = s.get("parsed_tokens", [])
            self._accumulate_sample_tokens(pid, toks, cnt)
            traits = s.get("traits", {})
            if "has_duplicates" in traits:
                cnt[f"trait:has_dup_{traits['has_duplicates']}"] += 0.5
            user_counts[pid] = (cnt, tot + len(toks))

        raw_user_vec = self._vectorize_probe_profile(user_counts, adaptive_smoothing=True)

        # Construct posterior-expected or manifold-mean reference profile for unobserved/sparse probes
        d = len(self.feature_keys)
        ref_vec = np.zeros(d, dtype=np.float64)
        if posteriors and self.model_base_vecs:
            tot_w = 0.0
            for m, w in posteriors.items():
                if m in self.model_base_vecs and w > 0:
                    ref_vec += float(w) * self.model_base_vecs[m]
                    tot_w += float(w)
            if tot_w > 0:
                ref_vec /= tot_w
            elif self.pca.mean_ is not None:
                ref_vec = self.pca.mean_.copy()
        elif self.model_base_vecs:
            ref_vec = np.mean(list(self.model_base_vecs.values()), axis=0)
        elif self.pca.mean_ is not None:
            ref_vec = self.pca.mean_.copy()

        user_vec = np.zeros(d, dtype=np.float64)
        for pid, (st, ed) in self.probe_slices.items():
            _, tot = user_counts.get(pid, ({}, 0.0))
            if tot <= 0.0:
                # Impute unobserved probe block with posterior/manifold expectation
                sub_vec = ref_vec[st:ed].copy()
            else:
                # Blend finite-sample empirical Hellinger slice with reference slice
                gamma = (tot / (tot + 15.0)) if posteriors else 1.0
                sub_vec = gamma * raw_user_vec[st:ed] + (1.0 - gamma) * ref_vec[st:ed]
            sub_norm = np.linalg.norm(sub_vec)
            if sub_norm > 0:
                user_vec[st:ed] = sub_vec / sub_norm

        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec = user_vec / norm

        pt_2d = self.pca.transform(user_vec.reshape(1, -1))[0]
        return [round(float(pt_2d[0]), 3), round(float(pt_2d[1]), 3)]

    def get_cluster_data(self, user_point: Optional[List[float]] = None) -> Dict[str, Any]:
        if not self.is_fitted or not hasattr(self, "_cached_base_clusters"):
            self.fit()

        clusters = getattr(self, "_cached_base_clusters", None)
        if clusters is None:
            clusters = self._build_clusters_payload()

        return {
            "clusters": clusters,
            "user_point": user_point
        }
