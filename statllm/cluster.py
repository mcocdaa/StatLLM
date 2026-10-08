"""
StatLLM Dimension Reduction & Clustering Visualizer for Array Probes.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Tuple, Optional

from statllm.probes import PROBES
from statllm.database import Database


class ClusterProjector:
    def __init__(self, db: Database):
        self.db = db
        self.pca: Optional[PCA] = None
        self.feature_keys: List[Tuple[str, str]] = []
        self.model_centers_2d: Dict[str, List[float]] = {}
        self.model_clouds_2d: Dict[str, List[List[float]]] = {}
        self.is_fitted = False

    def _build_feature_vocabulary(self) -> List[Tuple[str, str]]:
        feature_keys = []
        for pid, probe in PROBES.items():
            if probe.allowed_elements:
                for val in probe.allowed_elements:
                    feature_keys.append((pid, val))
            # Include behavioral trait markers
            feature_keys.append((pid, "trait:has_dup_True"))
            feature_keys.append((pid, "trait:has_dup_False"))
            feature_keys.append((pid, "INVALID"))
        return feature_keys

    def fit(self, n_points_per_model: Optional[int] = None):
        self.feature_keys = self._build_feature_vocabulary()
        d = len(self.feature_keys)
        key_to_idx = {k: i for i, k in enumerate(self.feature_keys)}

        models = self.db.list_models()
        model_names = [m["name"] for m in models] if models else []
        all_samples = self.db.get_all_samples()

        # If we have real empirical samples in the database and n_points_per_model is not explicitly forcing synthetic:
        if len(all_samples) >= 10 and n_points_per_model is None:
            X = []
            sample_model_names = []
            for s in all_samples:
                if not s.get("is_valid", 1):
                    continue
                vec = np.zeros(d, dtype=np.float64)
                pid = s["probe_id"]
                try:
                    import json
                    tokens = json.loads(s["parsed_value"])
                except Exception:
                    tokens = []
                for tok in tokens:
                    idx = key_to_idx.get((pid, tok))
                    if idx is not None:
                        vec[idx] += 1.0
                has_dup = len(tokens) != len(set(tokens))
                dup_key = (pid, f"trait:has_dup_{has_dup}")
                if dup_key in key_to_idx:
                    vec[key_to_idx[dup_key]] += 0.5
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                X.append(vec)
                sample_model_names.append(s["model_name"])

            X = np.array(X)
            self.pca = PCA(n_components=2, random_state=2026)
            projected = self.pca.fit_transform(X)

            self.model_clouds_2d = {m: [] for m in model_names}
            for name, pt in zip(sample_model_names, projected):
                if name in self.model_clouds_2d:
                    self.model_clouds_2d[name].append([round(float(pt[0]), 3), round(float(pt[1]), 3)])

            self.model_centers_2d = {}
            for name, pts in self.model_clouds_2d.items():
                if pts:
                    center = np.mean(pts, axis=0)
                    self.model_centers_2d[name] = [round(float(center[0]), 3), round(float(center[1]), 3)]
                else:
                    self.model_centers_2d[name] = [0.0, 0.0]

            self.is_fitted = True
            self._cached_base_clusters = self._build_clusters_payload()
            return

        # Fallback for synthetic / sparse test environments
        matrix = self.db.get_all_model_probe_counts()
        if not model_names:
            model_names = list(matrix.keys())

        if len(model_names) < 2:
            return

        all_points = []
        self.model_clouds_2d = {}
        self.model_centers_2d = {}
        pts_count = n_points_per_model if n_points_per_model is not None else 40

        rng = np.random.default_rng(2026)

        for m in model_names:
            m_data = matrix.get(m, {})
            base_vec = np.zeros(d, dtype=np.float64)
            for i, (pid, val) in enumerate(self.feature_keys):
                counts_dict, tot = m_data.get(pid, ({}, 0.0))
                c = counts_dict.get(val, 0.0)
                v_size = PROBES[pid].element_vocab_size if pid in PROBES else 10
                base_vec[i] = (c + 0.5) / (tot + 0.5 * v_size)

            for _ in range(pts_count):
                noise = rng.normal(0, 0.03, size=d)
                sample_vec = np.clip(base_vec + noise, 0.001, 1.0)
                sample_vec = sample_vec / (np.linalg.norm(sample_vec) + 1e-9)
                all_points.append(sample_vec)

        all_points = np.array(all_points)
        self.pca = PCA(n_components=2, random_state=2026)
        projected = self.pca.fit_transform(all_points)

        idx = 0
        for m in model_names:
            pts = projected[idx : idx + pts_count]
            idx += pts_count
            self.model_clouds_2d[m] = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in pts]
            center = np.mean(pts, axis=0)
            self.model_centers_2d[m] = [round(float(center[0]), 3), round(float(center[1]), 3)]

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
                "points": points
            })
        return clusters

    def project_user_submission(self, parsed_submissions: List[Dict[str, Any]]) -> Optional[List[float]]:
        if not self.is_fitted or self.pca is None or not self.feature_keys:
            self.fit()
            if not self.is_fitted or self.pca is None:
                return None

        d = len(self.feature_keys)
        user_vec = np.zeros(d, dtype=np.float64)

        user_counts = {}
        for s in parsed_submissions:
            pid = s["probe_id"]
            for tok in s.get("parsed_tokens", []):
                key = (pid, tok)
                user_counts[key] = user_counts.get(key, 0) + 1
            traits = s.get("traits", {})
            if "has_duplicates" in traits:
                key = (pid, f"trait:has_dup_{traits['has_duplicates']}")
                user_counts[key] = user_counts.get(key, 0) + 1

        for i, key in enumerate(self.feature_keys):
            c = user_counts.get(key, 0)
            user_vec[i] = c + 0.1

        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-9)
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
