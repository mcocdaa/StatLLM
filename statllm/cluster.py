"""
StatLLM Dimension Reduction & Clustering Visualizer.

Projects high-dimensional discrete probability distributions into a 2D space (via PCA)
to visualize model clusters and map the user's test sample onto the cluster map ("★ You Are Here").
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
        self.feature_keys: List[Tuple[str, str]] = []  # List of (probe_id, value)
        self.model_centers_2d: Dict[str, List[float]] = {}
        self.model_clouds_2d: Dict[str, List[List[float]]] = {}
        self.is_fitted = False

    def _build_feature_vocabulary(self) -> List[Tuple[str, str]]:
        """
        Builds the global feature space across all probes.
        Each feature is a specific (probe_id, discrete_value).
        """
        feature_keys = []
        for pid, probe in PROBES.items():
            if probe.allowed_values:
                for val in probe.allowed_values:
                    feature_keys.append((pid, val))
            else:
                # For probes with dynamic values, extract distinct values from DB
                pass
            feature_keys.append((pid, "INVALID"))
        return feature_keys

    def fit(self, n_points_per_model: int = 40):
        """
        Fits a 2D PCA projector using Dirichlet-sampled virtual points
        from the empirical counts of each model in the database.
        """
        matrix = self.db.get_all_model_probe_counts()
        models = self.db.list_models()
        model_names = [m["name"] for m in models] if models else list(matrix.keys())

        if len(model_names) < 2:
            return

        self.feature_keys = self._build_feature_vocabulary()
        d = len(self.feature_keys)

        # Generate cloud points for each model to simulate empirical distribution variance
        all_points = []
        labels = []
        self.model_clouds_2d = {}
        self.model_centers_2d = {}

        rng = np.random.default_rng(42)

        for m in model_names:
            model_vecs = []
            m_data = matrix.get(m, {})
            
            # Base probability vector
            base_vec = np.zeros(d, dtype=np.float64)
            for i, (pid, val) in enumerate(self.feature_keys):
                counts_dict, tot = m_data.get(pid, ({}, 0.0))
                c = counts_dict.get(val, 0.0)
                v_size = PROBES[pid].vocab_size if pid in PROBES else 10
                base_vec[i] = (c + 0.5) / (tot + 0.5 * v_size)

            # Generate synthetic bootstrap variations around base distribution
            for _ in range(n_points_per_model):
                # Add Dirichlet/multinomial noise
                noise = rng.normal(0, 0.04, size=d)
                sample_vec = np.clip(base_vec + noise, 0.001, 1.0)
                # Normalize per probe
                sample_vec = sample_vec / (np.linalg.norm(sample_vec) + 1e-9)
                model_vecs.append(sample_vec)
                all_points.append(sample_vec)
                labels.append(m)

        all_points = np.array(all_points)

        # Fit PCA to 2 components
        self.pca = PCA(n_components=2, random_state=42)
        projected = self.pca.fit_transform(all_points)

        # Group projected 2D points by model
        idx = 0
        for m in model_names:
            pts = projected[idx : idx + n_points_per_model]
            idx += n_points_per_model
            self.model_clouds_2d[m] = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in pts]
            center = np.mean(pts, axis=0)
            self.model_centers_2d[m] = [round(float(center[0]), 3), round(float(center[1]), 3)]

        self.is_fitted = True

    def project_user_submission(self, parsed_submissions: List[Dict[str, Any]]) -> Optional[List[float]]:
        """
        Projects user's parsed submissions onto the 2D plane.
        """
        if not self.is_fitted or self.pca is None or not self.feature_keys:
            self.fit()
            if not self.is_fitted or self.pca is None:
                return None

        d = len(self.feature_keys)
        user_vec = np.zeros(d, dtype=np.float64)

        # Count frequencies in user submissions
        user_counts = {}
        for s in parsed_submissions:
            key = (s["probe_id"], s["parsed_value"])
            user_counts[key] = user_counts.get(key, 0) + 1

        for i, key in enumerate(self.feature_keys):
            c = user_counts.get(key, 0)
            # Add smoothing
            user_vec[i] = c + 0.1

        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-9)
        pt_2d = self.pca.transform(user_vec.reshape(1, -1))[0]
        return [round(float(pt_2d[0]), 3), round(float(pt_2d[1]), 3)]

    def get_cluster_data(self, user_point: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Returns full 2D cluster visualization payload.
        """
        if not self.is_fitted:
            self.fit()

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

        return {
            "clusters": clusters,
            "user_point": user_point
        }
