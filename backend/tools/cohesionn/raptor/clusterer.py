"""
RAPTOR Clusterer - UMAP dimensionality reduction + GMM clustering

Groups semantically similar chunks for hierarchical summarization.
Optimized for engineering domain content.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result from clustering operation"""

    cluster_assignments: List[int]  # Cluster ID for each input item
    n_clusters: int
    cluster_sizes: Dict[int, int]  # cluster_id -> num items
    reduced_embeddings: Optional[np.ndarray] = None  # UMAP reduced (for visualization)


class RaptorClusterer:
    """
    UMAP + GMM clustering for RAPTOR hierarchical summarization.

    Features:
    - UMAP dimensionality reduction (1024 -> 15 dims) for better clustering
    - Gaussian Mixture Model with automatic component selection
    - Target cluster size ~10 items for optimal summarization
    - Handles small datasets gracefully
    """

    # UMAP parameters optimized for 1024-dim embeddings
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1
    UMAP_N_COMPONENTS = 15  # Reduce from 1024 to 15 dims
    UMAP_METRIC = "cosine"

    # GMM parameters
    GMM_COVARIANCE_TYPE = "full"
    GMM_MAX_ITER = 200
    GMM_N_INIT = 5
    GMM_REG_COVAR = 1e-5  # Regularization for stability

    def __init__(
        self,
        target_cluster_size: int = 10,
        min_cluster_size: int = 3,
        max_cluster_size: int = 20,
        use_umap: bool = True,
        random_state: int = 42,
    ):
        """
        Args:
            target_cluster_size: Target number of items per cluster
            min_cluster_size: Minimum items to form a cluster
            max_cluster_size: Maximum items per cluster
            use_umap: Whether to use UMAP reduction (recommended)
            random_state: Random seed for reproducibility
        """
        self.target_cluster_size = target_cluster_size
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.use_umap = use_umap
        self.random_state = random_state

        self._umap_model = None
        self._gmm_model = None

    def cluster(
        self,
        embeddings: List[List[float]],
        n_clusters: Optional[int] = None,
    ) -> ClusterResult:
        """
        Cluster embeddings using UMAP + GMM.

        Args:
            embeddings: List of embedding vectors (1024-dim)
            n_clusters: Optional fixed number of clusters (auto-determined if None)

        Returns:
            ClusterResult with assignments and metadata
        """
        n_items = len(embeddings)

        if n_items == 0:
            return ClusterResult(
                cluster_assignments=[],
                n_clusters=0,
                cluster_sizes={},
            )

        # Handle very small datasets
        if n_items < self.min_cluster_size:
            # Everything in one cluster
            return ClusterResult(
                cluster_assignments=[0] * n_items,
                n_clusters=1,
                cluster_sizes={0: n_items},
            )

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Step 1: UMAP dimensionality reduction
        if self.use_umap and n_items > self.UMAP_N_NEIGHBORS:
            reduced = self._reduce_dimensions(embeddings_array)
        else:
            reduced = embeddings_array
            logger.debug(f"Skipping UMAP: n_items={n_items} <= n_neighbors={self.UMAP_N_NEIGHBORS}")

        # Step 2: Determine optimal number of clusters
        if n_clusters is None:
            n_clusters = self._estimate_n_clusters(n_items)

        # Clamp to valid range
        n_clusters = max(1, min(n_clusters, n_items // self.min_cluster_size))

        # Step 3: GMM clustering
        assignments = self._gmm_cluster(reduced, n_clusters)

        # Calculate cluster sizes
        cluster_sizes = {}
        for cluster_id in assignments:
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        logger.info(f"Clustered {n_items} items into {len(cluster_sizes)} clusters")
        logger.debug(f"Cluster sizes: {dict(sorted(cluster_sizes.items()))}")

        return ClusterResult(
            cluster_assignments=assignments,
            n_clusters=len(cluster_sizes),
            cluster_sizes=cluster_sizes,
            reduced_embeddings=reduced if self.use_umap else None,
        )

    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP dimensionality reduction."""
        try:
            import umap

            n_neighbors = min(self.UMAP_N_NEIGHBORS, len(embeddings) - 1)
            n_components = min(self.UMAP_N_COMPONENTS, len(embeddings) - 1)

            self._umap_model = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=self.UMAP_MIN_DIST,
                metric=self.UMAP_METRIC,
                random_state=self.random_state,
                low_memory=True,  # Memory efficient for large datasets
            )

            reduced = self._umap_model.fit_transform(embeddings)
            logger.debug(f"UMAP reduced {embeddings.shape} -> {reduced.shape}")
            return reduced

        except ImportError:
            logger.warning("umap-learn not installed, skipping dimensionality reduction")
            return embeddings
        except Exception as e:
            logger.warning(f"UMAP failed, using original embeddings: {e}")
            return embeddings

    def _estimate_n_clusters(self, n_items: int) -> int:
        """Estimate optimal number of clusters based on target size."""
        # Target: ~10 items per cluster
        estimated = max(1, n_items // self.target_cluster_size)

        # Clamp to reasonable range
        min_clusters = max(1, n_items // self.max_cluster_size)
        max_clusters = max(1, n_items // self.min_cluster_size)

        return max(min_clusters, min(estimated, max_clusters))

    def _gmm_cluster(self, data: np.ndarray, n_clusters: int) -> List[int]:
        """Apply Gaussian Mixture Model clustering."""
        try:
            from sklearn.mixture import GaussianMixture

            self._gmm_model = GaussianMixture(
                n_components=n_clusters,
                covariance_type=self.GMM_COVARIANCE_TYPE,
                max_iter=self.GMM_MAX_ITER,
                n_init=self.GMM_N_INIT,
                reg_covar=self.GMM_REG_COVAR,
                random_state=self.random_state,
            )

            assignments = self._gmm_model.fit_predict(data)
            return assignments.tolist()

        except ImportError:
            logger.warning("scikit-learn not installed, using simple assignment")
            return self._fallback_cluster(len(data), n_clusters)
        except Exception as e:
            logger.warning(f"GMM failed, using fallback: {e}")
            return self._fallback_cluster(len(data), n_clusters)

    def _fallback_cluster(self, n_items: int, n_clusters: int) -> List[int]:
        """Simple round-robin assignment as fallback."""
        return [i % n_clusters for i in range(n_items)]

    def cluster_by_similarity(
        self,
        embeddings: List[List[float]],
        threshold: float = 0.7,
    ) -> ClusterResult:
        """
        Alternative: cluster by cosine similarity threshold.

        Useful when you want natural clusters rather than fixed count.

        Args:
            embeddings: Embedding vectors
            threshold: Minimum similarity to be in same cluster

        Returns:
            ClusterResult
        """
        n_items = len(embeddings)

        if n_items == 0:
            return ClusterResult([], 0, {})

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings_array / norms

        # Compute similarity matrix
        similarity = np.dot(normalized, normalized.T)

        # Greedy clustering
        assigned = [-1] * n_items
        cluster_id = 0

        for i in range(n_items):
            if assigned[i] >= 0:
                continue

            # Start new cluster
            assigned[i] = cluster_id

            # Add similar items
            for j in range(i + 1, n_items):
                if assigned[j] < 0 and similarity[i, j] >= threshold:
                    assigned[j] = cluster_id

            cluster_id += 1

        # Calculate sizes
        cluster_sizes = {}
        for cid in assigned:
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

        return ClusterResult(
            cluster_assignments=assigned,
            n_clusters=cluster_id,
            cluster_sizes=cluster_sizes,
        )
