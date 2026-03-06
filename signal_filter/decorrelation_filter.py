"""
signal_filter/decorrelation_filter.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer D — Decorrelation & Redundancy Removal

Tujuan:
  Dari semua fitur yang sudah lolos A+B+C, buang yang redundant
  (highly correlated satu sama lain). Pertahankan yang terbaik
  dari setiap cluster berdasarkan IC IR.

Pendekatan:
  1. Hitung correlation matrix antar semua fitur yang lolos
  2. Hierarchical clustering berdasarkan |correlation|
  3. Dari setiap cluster, ambil representative (IC IR tertinggi)

Ini penting karena:
  - Fitur yang highly correlated memberikan informasi yang sama ke model
  - Memasukkan keduanya ke Atomic hanya buang waktu komputasi
  - Lebih parah: bisa menyebabkan false confidence di ensemble
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


@dataclass
class DecorrelationConfig:
    max_correlation: float = 0.80     # Threshold korelasi untuk clustering
    method: str = "average"           # Linkage method: average, complete, ward
    min_cluster_ic_ir: float = 0.0    # Min IC IR untuk representative (biasanya sudah difilter)


@dataclass
class DecorrelationResult:
    # Fitur yang terpilih (representative per cluster)
    selected_features: List[str] = field(default_factory=list)

    # Mapping: feature → cluster_id
    cluster_map: Dict[str, int] = field(default_factory=dict)

    # Mapping: cluster_id → list semua fitur di cluster
    clusters: Dict[int, List[str]] = field(default_factory=dict)

    # Fitur yang di-drop (bukan representative)
    dropped_features: List[str] = field(default_factory=list)

    # Stats
    n_clusters: int = 0
    n_input: int = 0
    n_output: int = 0


class DecorrelationFilter:
    """
    Layer D: Decorrelation & Redundancy Removal.

    Input : fitur yang sudah lolos A+B+C beserta IC IR score-nya
    Output: subset fitur non-redundant, dipilih berdasarkan IC IR
    """

    def __init__(self, config: Optional[DecorrelationConfig] = None):
        self.config = config or DecorrelationConfig()

    def run(
        self,
        df: pd.DataFrame,
        feature_list: List[str],
        ic_ir_scores: Optional[Dict[str, float]] = None
    ) -> Tuple[List[str], DecorrelationResult]:
        """
        Run decorrelation filter.

        Args:
            df            : DataFrame dengan semua kolom fitur
            feature_list  : fitur yang sudah lolos filter sebelumnya
            ic_ir_scores  : dict {feature: ic_ir} untuk ranking dalam cluster
                           Jika None, semua dianggap equal → ambil pertama

        Returns:
            selected : list fitur yang terpilih
            result   : DecorrelationResult dengan detail clustering
        """
        cfg = self.config

        # Validasi kolom tersedia
        valid_features = [f for f in feature_list if f in df.columns]
        if len(valid_features) <= 1:
            return valid_features, DecorrelationResult(
                selected_features=valid_features,
                n_input=len(valid_features),
                n_output=len(valid_features),
                n_clusters=len(valid_features)
            )

        # Default IC IR: semua 0 (equal)
        if ic_ir_scores is None:
            ic_ir_scores = {f: 0.0 for f in valid_features}

        # ── Correlation matrix ───────────────────────────────────
        feat_data = df[valid_features].dropna()
        if len(feat_data) < 30:
            # Tidak cukup data untuk korelasi — return semua
            return valid_features, DecorrelationResult(
                selected_features=valid_features,
                n_input=len(valid_features),
                n_output=len(valid_features),
                n_clusters=len(valid_features)
            )

        corr_matrix = feat_data.corr(method="spearman").abs()

        # Pastikan diagonal = 1, fill NaN dengan 0
        corr_arr = corr_matrix.values.copy()
        np.fill_diagonal(corr_arr, 1.0)
        corr_matrix = pd.DataFrame(corr_arr, index=corr_matrix.index, columns=corr_matrix.columns)
        corr_matrix = corr_matrix.fillna(0.0)

        # ── Hierarchical clustering ───────────────────────────────
        # Distance = 1 - |correlation|
        dist_matrix = 1.0 - corr_matrix.values
        np.fill_diagonal(dist_matrix, 0.0)

        # Pastikan distance matrix valid (symmetri, non-negative)
        dist_matrix = np.clip(dist_matrix, 0, None)
        dist_condensed = squareform(dist_matrix, checks=False)

        Z = linkage(dist_condensed, method=cfg.method)

        # Cut tree pada threshold distance = 1 - max_correlation
        threshold = 1.0 - cfg.max_correlation
        cluster_labels = fcluster(Z, t=threshold, criterion="distance")

        # ── Build clusters ────────────────────────────────────────
        clusters: Dict[int, List[str]] = {}
        cluster_map: Dict[str, int] = {}

        for feat, cluster_id in zip(valid_features, cluster_labels):
            cid = int(cluster_id)
            cluster_map[feat] = cid
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(feat)

        # ── Select representative per cluster ─────────────────────
        selected = []
        dropped = []

        for cid, members in clusters.items():
            if len(members) == 1:
                selected.append(members[0])
                continue

            # Pilih yang IC IR tertinggi
            best = max(members, key=lambda f: ic_ir_scores.get(f, 0.0))
            selected.append(best)
            dropped.extend([f for f in members if f != best])

        result = DecorrelationResult(
            selected_features=selected,
            cluster_map=cluster_map,
            clusters=clusters,
            dropped_features=dropped,
            n_clusters=len(clusters),
            n_input=len(valid_features),
            n_output=len(selected)
        )

        return selected, result

    def summary_df(self, result: DecorrelationResult, ic_ir_scores: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Buat DataFrame ringkasan clustering untuk logging/display.
        """
        rows = []
        for cid, members in result.clusters.items():
            rep = [f for f in result.selected_features if result.cluster_map.get(f) == cid]
            representative = rep[0] if rep else members[0]
            for feat in members:
                rows.append({
                    "cluster_id":    cid,
                    "feature":       feat,
                    "is_selected":   feat in result.selected_features,
                    "representative": representative,
                    "ic_ir":         ic_ir_scores.get(feat, np.nan) if ic_ir_scores else np.nan,
                    "cluster_size":  len(members)
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["cluster_id", "is_selected"], ascending=[True, False])
        return df
