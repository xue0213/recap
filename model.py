from torch import nn
import torch.nn.functional as F
import torch
import hashlib
import os

from large_graph_knn import select_faiss_ivfpq_candidates


class EgoCluster(nn.Module):
    """Residual community assignment module used in RECAP.

    This module implements the core equations:
    - Assignment: H = softmax(EW)
    - Loss: L_total = L_con + lambda_H * L_H + lambda_E * L_var
    - Score: s = norm(s_adhesion) + beta * norm(s_context), where s_context
      measures community-distribution inconsistency with KNN ego neighbors.
    """

    def __init__(
        self,
        embed_dim: int,
        num_clusters: int = 10,
        knn_k: int = 5,
        tau_s: float = 0.2,
        tau_c: float = 1.0,
        cluster_init_gain: float | None = None,
        tau_e: float = 1.0,
        beta: float = 0.1,
        lambda_H: float = 1.0,
        lambda_bal: float = 0.1,
        lambda_E: float = 0.0,
        lambda_usage_entropy: float = 0.0,
        assignment_entropy_lower: float | None = None,
        assignment_entropy_upper: float | None = None,
        usage_entropy_lower: float | None = None,
        usage_entropy_upper: float | None = None,
        gamma: float = 0.01,
        eps: float = 1e-8,
        sim_block_size: int = 1024,
        knn_cache_enabled: bool = True,
        knn_cache_dir: str = "./knn_cache",
        knn_search_dtype: str = "auto",
        ann_large_datasets=("tsocial", "dgraphfin"),
        ann_backend: str = "faiss_ivfpq",
        ann_nlist: int = 4096,
        ann_nprobe: int = 16,
        ann_pq_m: int = 16,
        ann_train_size: int = 262_144,
        ann_query_batch_size: int = 4_096,
        ann_add_batch_size: int = 262_144,
        ann_rerank_factor: int = 32,
        ann_max_rerank_candidates: int = 256,
        ann_score_batch_size: int = 2_048,
        ann_seed: int = 0,
    ):
        super().__init__()
        self.knn_k = knn_k
        self.num_clusters = num_clusters

        self.tau_s = tau_s
        self.tau_c = tau_c
        self.cluster_init_gain = cluster_init_gain
        self.tau_e = tau_e
        self.beta = beta

        self.lambda_H = lambda_H
        self.lambda_bal = lambda_bal
        self.lambda_E = lambda_E
        self.lambda_usage_entropy = lambda_usage_entropy
        self.assignment_entropy_lower = assignment_entropy_lower
        self.assignment_entropy_upper = assignment_entropy_upper
        self.usage_entropy_lower = usage_entropy_lower
        self.usage_entropy_upper = usage_entropy_upper
        self.gamma = gamma
        self.eps = eps
        self.sim_block_size = sim_block_size
        self.knn_cache_enabled = knn_cache_enabled
        self.knn_cache_dir = knn_cache_dir
        self.knn_search_dtype = knn_search_dtype
        self.ann_large_datasets = {
            str(name).lower() for name in ann_large_datasets
        }
        self.ann_backend = ann_backend
        self.ann_nlist = ann_nlist
        self.ann_nprobe = ann_nprobe
        self.ann_pq_m = ann_pq_m
        self.ann_train_size = ann_train_size
        self.ann_query_batch_size = ann_query_batch_size
        self.ann_add_batch_size = ann_add_batch_size
        self.ann_rerank_factor = ann_rerank_factor
        self.ann_max_rerank_candidates = ann_max_rerank_candidates
        self.ann_score_batch_size = ann_score_batch_size
        self.ann_seed = ann_seed

        self.W = nn.Linear(embed_dim, num_clusters, bias=False)
        if self.cluster_init_gain is not None:
            nn.init.xavier_uniform_(self.W.weight, gain=float(self.cluster_init_gain))
        self._knn_cache: dict = {}

    def _resolve_knn_search_dtype(self, device: torch.device) -> torch.dtype:
        requested = str(self.knn_search_dtype).lower()
        if device.type != "cuda":
            return torch.float32

        if requested in {"float32", "fp32"}:
            return torch.float32
        if requested in {"float16", "fp16", "half"}:
            return torch.float16
        if requested in {"bfloat16", "bf16"}:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if requested == "auto":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        raise ValueError(
            "knn_search_dtype must be one of auto, float32, float16, bfloat16; "
            f"got {self.knn_search_dtype!r}"
        )

    def _disk_cache_path(self, full_key: tuple) -> str | None:
        if not self.knn_cache_enabled or not self.knn_cache_dir:
            return None

        cache_key = full_key[1] if len(full_key) > 1 else None
        if isinstance(cache_key, tuple) and cache_key and cache_key[0] == "runtime":
            return None

        digest = hashlib.sha1(repr(full_key).encode("utf-8")).hexdigest()
        return os.path.join(self.knn_cache_dir, f"knn_{digest}.pt")

    def _load_knn_from_disk(self, full_key: tuple) -> torch.Tensor | None:
        cache_path = self._disk_cache_path(full_key)
        if cache_path is None or not os.path.exists(cache_path):
            return None

        try:
            payload = torch.load(cache_path, map_location="cpu")
            if payload.get("key_repr") != repr(full_key):
                return None
            topk_idx = payload.get("topk_idx")
            if not torch.is_tensor(topk_idx):
                return None
            return topk_idx.long().cpu()
        except Exception as exc:
            print(f"KNN磁盘缓存读取失败，将重新计算: {cache_path} ({exc})")
            return None

    def _save_knn_to_disk(self, full_key: tuple, topk_idx: torch.Tensor) -> None:
        cache_path = self._disk_cache_path(full_key)
        if cache_path is None:
            return

        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            torch.save(
                {
                    "version": "recap_knn_candidates_v1",
                    "key_repr": repr(full_key),
                    "topk_idx": topk_idx.detach().long().cpu(),
                },
                tmp_path,
            )
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            print(f"KNN磁盘缓存写入失败，继续使用内存缓存: {cache_path} ({exc})")

    def _cosine_similarity(self, E: torch.Tensor) -> torch.Tensor:
        E_norm = F.normalize(E, p=2, dim=1)
        return torch.mm(E_norm, E_norm.t())

    @torch.no_grad()
    def _select_knn_candidates(self, E_init: torch.Tensor) -> torch.Tensor:
        """Select fixed KNN candidates from initial residual features."""
        N = E_init.shape[0]
        k = min(self.knn_k, max(N - 1, 0))
        if k == 0:
            return torch.empty(N, 0, dtype=torch.long, device=E_init.device)

        block = max(1, min(self.sim_block_size, N))
        search_dtype = self._resolve_knn_search_dtype(E_init.device)
        E_init_norm = F.normalize(E_init.detach().float(), p=2, dim=1)
        if search_dtype != torch.float32:
            E_init_norm = E_init_norm.to(search_dtype)
        all_topk_idx = []

        for start in range(0, N, block):
            end = min(start + block, N)
            row_idx = torch.arange(start, end, device=E_init.device)
            sim_block = torch.mm(E_init_norm[start:end], E_init_norm.t())
            sim_block[torch.arange(end - start, device=E_init.device), row_idx] = float("-inf")
            _, topk_idx = sim_block.topk(k, dim=1)
            all_topk_idx.append(topk_idx)

        return torch.cat(all_topk_idx, dim=0)

    def _dataset_uses_ann(self, cache_key: tuple | None) -> bool:
        return (
            isinstance(cache_key, tuple)
            and len(cache_key) > 1
            and cache_key[0] == "dataset"
            and str(cache_key[1]).lower() in self.ann_large_datasets
        )

    @torch.no_grad()
    def _select_approximate_knn_candidates(
        self, E_init: torch.Tensor
    ) -> torch.Tensor:
        if self.ann_backend != "faiss_ivfpq":
            raise ValueError(f"Unsupported ANN backend: {self.ann_backend!r}")
        return select_faiss_ivfpq_candidates(
            E_init,
            self.knn_k,
            nlist=self.ann_nlist,
            nprobe=self.ann_nprobe,
            pq_m=self.ann_pq_m,
            train_size=self.ann_train_size,
            query_batch_size=self.ann_query_batch_size,
            add_batch_size=self.ann_add_batch_size,
            rerank_factor=self.ann_rerank_factor,
            max_rerank_candidates=self.ann_max_rerank_candidates,
            seed=self.ann_seed,
        )

    def _get_knn_candidates(
        self, E_init: torch.Tensor, cache_key: tuple | None = None
    ) -> torch.Tensor:
        if cache_key is None:
            return self._select_knn_candidates(E_init)

        device = E_init.device
        search_dtype = self._resolve_knn_search_dtype(device)
        use_ann = self._dataset_uses_ann(cache_key)
        if use_ann:
            full_key = (
                "recap_ann_knn_candidates_v1",
                cache_key,
                E_init.shape[0],
                E_init.shape[1],
                self.knn_k,
                self.ann_backend,
                self.ann_nlist,
                self.ann_nprobe,
                self.ann_pq_m,
                self.ann_train_size,
                self.ann_rerank_factor,
                self.ann_max_rerank_candidates,
                self.ann_seed,
            )
        else:
            full_key = (
                "recap_knn_candidates_v1",
                cache_key,
                E_init.shape[0],
                E_init.shape[1],
                self.knn_k,
                str(search_dtype).replace("torch.", ""),
            )
        cached = self._knn_cache.get(full_key)
        if cached is None:
            cached = self._load_knn_from_disk(full_key)
            if cached is not None:
                self._knn_cache[full_key] = cached
        if cached is None:
            if use_ann:
                cached = self._select_approximate_knn_candidates(E_init)
            else:
                cached = self._select_knn_candidates(E_init)
            cached = cached.detach().cpu()
            self._knn_cache[full_key] = cached
            self._save_knn_to_disk(full_key, cached)
        return cached.to(device)

    def build_ego_graph(
        self,
        E: torch.Tensor,
        E_init: torch.Tensor | None = None,
        cache_key: tuple | None = None,
    ):
        """Build the paper-aligned symmetrized residual similarity graph.

        KNN candidate sets are selected once from initial residual features
        (E_init). Per training step, soft weights are computed only on the
        fixed N x k candidate edge set, so the differentiable graph update is
        O(nkd) instead of recomputing dense N x N similarities.
        Returns:
            edge_index: LongTensor (2, E)
            edge_weight: FloatTensor (E,)
        """
        N = E.shape[0]
        if E_init is None:
            E_init = E.detach()

        topk_idx = self._get_knn_candidates(E_init, cache_key=cache_key)
        k = topk_idx.shape[1]
        if k == 0:
            return (
                torch.empty(2, 0, dtype=torch.long, device=E.device),
                torch.empty(0, dtype=E.dtype, device=E.device),
            )

        E_norm = F.normalize(E, p=2, dim=1)

        scale = max(self.tau_s, self.eps)
        candidate_emb = E_norm[topk_idx]  # (N, k, d)
        topk_scores = (candidate_emb * E_norm.unsqueeze(1)).sum(dim=2)
        topk_weights = F.softmax(topk_scores / scale, dim=1)

        rows = torch.arange(N, device=E.device).unsqueeze(1).expand(N, k).reshape(-1)
        cols = topk_idx.reshape(-1)
        weights = topk_weights.reshape(-1)

        # Paper Eq. 12: S <- (S + S^T) / 2. Duplicated reciprocal edges are
        # intentionally kept; index_add in the loss sums them as sparse entries.
        edge_row = torch.cat([rows, cols])
        edge_col = torch.cat([cols, rows])
        edge_weight = torch.cat([weights, weights]) * 0.5
        edge_index = torch.stack([edge_row, edge_col], dim=0)
        return edge_index, edge_weight

    def cluster(self, E: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.W(E) / max(self.tau_c, self.eps), dim=1)

    def _compute_con_loss(
        self, H: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        if edge_weight.numel() == 0:
            return H.new_tensor(0.0)

        row, col = edge_index
        # SH_i = sum_j S_ij * H_j, computed from sparse edge entries.
        SH = torch.zeros_like(H)
        SH.index_add_(0, row, H[col] * edge_weight.unsqueeze(-1))

        deg = H.new_zeros(H.shape[0])
        deg.index_add_(0, row, edge_weight)
        DSH = deg.unsqueeze(1) * H
        num = (H * (DSH - SH)).sum()
        den = (H * DSH).sum() + self.eps
        return num / den

    def _compute_H_loss(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entropy = -(H * torch.log(H + self.eps)).sum(dim=1).mean()
        log_c = torch.log(H.new_tensor(max(H.shape[1], 2)))

        if self.assignment_entropy_lower is None and self.assignment_entropy_upper is None:
            l_sharp = entropy
        else:
            entropy_norm = entropy / log_c.clamp(min=self.eps)
            l_sharp = H.new_tensor(0.0)
            if self.assignment_entropy_upper is not None:
                l_sharp = l_sharp + F.relu(
                    entropy_norm - float(self.assignment_entropy_upper)
                )
            if self.assignment_entropy_lower is not None:
                l_sharp = l_sharp + F.relu(
                    float(self.assignment_entropy_lower) - entropy_norm
                )
            l_sharp = l_sharp * log_c

        h_bar = H.mean(dim=0)
        l_bal = (h_bar * torch.log(h_bar + self.eps)).sum()

        l_usage = H.new_tensor(0.0)
        if self.usage_entropy_lower is not None or self.usage_entropy_upper is not None:
            pi_entropy = -(h_bar * torch.log(h_bar + self.eps)).sum()
            pi_entropy_norm = pi_entropy / log_c.clamp(min=self.eps)
            if self.usage_entropy_upper is not None:
                l_usage = l_usage + F.relu(
                    pi_entropy_norm - float(self.usage_entropy_upper)
                )
            if self.usage_entropy_lower is not None:
                l_usage = l_usage + F.relu(
                    float(self.usage_entropy_lower) - pi_entropy_norm
                )
            l_usage = l_usage * log_c

        l_H = l_sharp + self.lambda_bal * l_bal + self.lambda_usage_entropy * l_usage
        return l_H, l_sharp, l_bal

    def _compute_var_loss(self, E: torch.Tensor) -> torch.Tensor:
        std_per_dim = E.std(dim=0, unbiased=False)
        return F.relu(self.gamma - std_per_dim).mean()

    def compute_losses(
        self,
        E: torch.Tensor,
        E_init: torch.Tensor | None = None,
        cache_key: tuple | None = None,
    ) -> torch.Tensor:
        """Compute RECAP training loss: L_con + lambda_H * L_H + lambda_E * L_var."""
        edge_index, edge_weight = self.build_ego_graph(E, E_init=E_init, cache_key=cache_key)
        H = self.cluster(E)

        l_con = self._compute_con_loss(H, edge_index, edge_weight)
        l_H, l_sharp, l_bal = self._compute_H_loss(H)
        l_var = self._compute_var_loss(E) if self.lambda_E != 0 else E.new_tensor(0.0)

        total_loss = l_con + self.lambda_H * l_H + self.lambda_E * l_var

        if hasattr(self, "_debug_losses") and self._debug_losses:
            print(
                "  Loss breakdown: "
                f"L_con={l_con:.6f}, "
                f"L_sharp={l_sharp:.6f}, "
                f"L_bal={l_bal:.6f}, "
                f"L_H={l_H:.6f} (x{self.lambda_H}={self.lambda_H * l_H:.6f}), "
                f"L_var={l_var:.6f} (x{self.lambda_E}={self.lambda_E * l_var:.6f}), "
                f"Total={total_loss:.6f}"
            )

        return total_loss

    def _compute_context_score(
        self,
        H: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_weight.numel() == 0:
            return H.new_zeros(H.shape[0]), H

        row, col = edge_index
        neighbor_H = H.new_zeros(H.shape)
        neighbor_H.index_add_(0, row, edge_weight.unsqueeze(1) * H[col])

        deg = H.new_zeros(H.shape[0])
        deg.index_add_(0, row, edge_weight)
        neighbor_H = neighbor_H / deg.clamp(min=self.eps).unsqueeze(1)
        neighbor_H = neighbor_H / neighbor_H.sum(dim=1, keepdim=True).clamp(min=self.eps)

        midpoint = 0.5 * (H + neighbor_H)
        kl_self = (H * (torch.log(H + self.eps) - torch.log(midpoint + self.eps))).sum(dim=1)
        kl_neighbor = (
            neighbor_H
            * (torch.log(neighbor_H + self.eps) - torch.log(midpoint + self.eps))
        ).sum(dim=1)
        js = 0.5 * (kl_self + kl_neighbor)
        js = js / torch.log(H.new_tensor(2.0))
        return js, neighbor_H

    def _compute_ann_context_score(
        self,
        E: torch.Tensor,
        H: torch.Tensor,
        topk_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the symmetrized KNN context score in bounded blocks."""
        num_nodes, k = topk_idx.shape
        if k == 0:
            return H.new_zeros(num_nodes), H

        E_norm = F.normalize(E, p=2, dim=1)
        neighbor_H = H.new_zeros(H.shape)
        degree = H.new_zeros(num_nodes)
        scale = max(self.tau_s, self.eps)
        block = max(1, int(self.ann_score_batch_size))
        for start in range(0, num_nodes, block):
            stop = min(start + block, num_nodes)
            cols = topk_idx[start:stop]
            candidate_emb = E_norm[cols]
            scores = (
                candidate_emb * E_norm[start:stop].unsqueeze(1)
            ).sum(dim=2)
            weights = F.softmax(scores / scale, dim=1) * 0.5

            neighbor_H[start:stop] += (
                H[cols] * weights.unsqueeze(-1)
            ).sum(dim=1)
            degree[start:stop] += weights.sum(dim=1)

            flat_cols = cols.reshape(-1)
            reverse_values = (
                H[start:stop].unsqueeze(1) * weights.unsqueeze(-1)
            ).reshape(-1, H.shape[1])
            neighbor_H.index_add_(0, flat_cols, reverse_values)
            degree.index_add_(0, flat_cols, weights.reshape(-1))

        neighbor_H = neighbor_H / degree.clamp(min=self.eps).unsqueeze(1)
        neighbor_H = neighbor_H / neighbor_H.sum(
            dim=1, keepdim=True
        ).clamp(min=self.eps)
        midpoint = 0.5 * (H + neighbor_H)
        kl_self = (
            H * (torch.log(H + self.eps) - torch.log(midpoint + self.eps))
        ).sum(dim=1)
        kl_neighbor = (
            neighbor_H
            * (
                torch.log(neighbor_H + self.eps)
                - torch.log(midpoint + self.eps)
            )
        ).sum(dim=1)
        js = 0.5 * (kl_self + kl_neighbor)
        return js / torch.log(H.new_tensor(2.0)), neighbor_H

    def _compute_adhesion_raw_chunked(
        self, E: torch.Tensor, H: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Compute centroid distances without materializing N x C x D."""
        output = E.new_empty(E.shape[0])
        z_norm = (Z * Z).sum(dim=1).unsqueeze(0)
        block = max(1, int(self.ann_score_batch_size))
        for start in range(0, E.shape[0], block):
            stop = min(start + block, E.shape[0])
            current = E[start:stop]
            dist_sq = (
                (current * current).sum(dim=1, keepdim=True)
                + z_norm
                - 2.0 * torch.mm(current, Z.t())
            ).clamp_min_(0)
            output[start:stop] = (
                H[start:stop] * dist_sq
            ).sum(dim=1) / max(self.tau_e, self.eps)
        return output

    def _compute_ann_score_components(
        self,
        E: torch.Tensor,
        E_init: torch.Tensor,
        cache_key: tuple,
    ) -> dict:
        H = self.cluster(E)
        topk_idx = self._get_knn_candidates(E_init, cache_key=cache_key)
        denom = H.sum(dim=0).clamp(min=self.eps)
        Z = torch.mm(H.t(), E) / denom.unsqueeze(1)
        s_adhesion_raw = self._compute_adhesion_raw_chunked(E, H, Z)
        s_scale_raw, neighbor_H = self._compute_ann_context_score(
            E, H, topk_idx
        )
        s_adhesion = self._standardize_score(s_adhesion_raw)
        s_scale = self._standardize_score(s_scale_raw)
        return {
            "total": s_adhesion + self.beta * s_scale,
            "adhesion": s_adhesion,
            "scale": s_scale,
            "adhesion_raw": s_adhesion_raw,
            "scale_raw": s_scale_raw,
            "neighbor_context": neighbor_H,
        }

    def compute_score_components(
        self,
        E: torch.Tensor,
        E_init: torch.Tensor | None = None,
        cache_key: tuple | None = None,
    ) -> dict:
        """Compute decomposed score components for each node."""
        if self._dataset_uses_ann(cache_key):
            if E_init is None:
                E_init = E.detach()
            return self._compute_ann_score_components(E, E_init, cache_key)

        H = self.cluster(E)
        edge_index, edge_weight = self.build_ego_graph(E, E_init=E_init, cache_key=cache_key)

        denom = H.sum(dim=0).clamp(min=self.eps)
        Z = torch.mm(H.t(), E) / denom.unsqueeze(1)

        diff = E.unsqueeze(1) - Z.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=2)

        s_adhesion_raw = (H * dist_sq).sum(dim=1) / max(self.tau_e, self.eps)
        s_scale_raw, neighbor_H = self._compute_context_score(H, edge_index, edge_weight)

        s_adhesion = self._standardize_score(s_adhesion_raw)
        s_scale = self._standardize_score(s_scale_raw)

        total = s_adhesion + self.beta * s_scale
        return {
            "total": total,
            "adhesion": s_adhesion,
            "scale": s_scale,
            "adhesion_raw": s_adhesion_raw,
            "scale_raw": s_scale_raw,
            "neighbor_context": neighbor_H,
        }

    def compute_scores(
        self,
        E: torch.Tensor,
        E_init: torch.Tensor | None = None,
        cache_key: tuple | None = None,
    ) -> torch.Tensor:
        """Compute anomaly score from adhesion and KNN community inconsistency."""
        return self.compute_score_components(E, E_init=E_init, cache_key=cache_key)["total"]

    def _standardize_score(self, score: torch.Tensor) -> torch.Tensor:
        return (score - score.mean()) / score.std(unbiased=False).clamp(min=self.eps)


class recap(nn.Module):
    def __init__(
        self,
        dims,
        h_feats=32,
        num_layers=2,
        dropout_rate=0,
        activation="ReLU",
        num_hops=4,
        num_views=1,
        num_clusters=10,
        knn_k=5,
        **kwargs,
    ):
        super(recap, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = num_hops
        self.num_views = 1
        self.num_layers = num_layers

        if num_layers > 0:
            self.layers.append(nn.Linear(dims, h_feats))
            for _ in range(1, num_layers):
                self.layers.append(nn.Linear(h_feats, h_feats))

        self.embed_dim = (h_feats if num_layers > 0 else dims) * num_hops
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.num_clusters = num_clusters
        self.knn_k = knn_k

        self.tau_s = kwargs.get("tau_s", 0.2)
        self.tau_c = kwargs.get("tau_c", 1.0)
        self.cluster_init_gain = kwargs.get("cluster_init_gain", None)
        self.tau_e = kwargs.get("tau_e", 1.0)
        self.beta = kwargs.get("beta", 0.1)

        # RECAP paper-aligned losses
        self.lambda_H = kwargs.get("lambda_H", kwargs.get("lambda_ortho", 1.0))
        self.lambda_bal = kwargs.get("lambda_bal", kwargs.get("lambda_min_usage", 0.1))
        self.lambda_E = kwargs.get("lambda_E", kwargs.get("lambda_diversity", 0.0))
        self.lambda_usage_entropy = kwargs.get("lambda_usage_entropy", 0.0)
        self.assignment_entropy_lower = kwargs.get("assignment_entropy_lower", None)
        self.assignment_entropy_upper = kwargs.get("assignment_entropy_upper", None)
        self.usage_entropy_lower = kwargs.get("usage_entropy_lower", None)
        self.usage_entropy_upper = kwargs.get("usage_entropy_upper", None)
        self.gamma = kwargs.get("gamma", kwargs.get("min_cluster_ratio", 0.01))
        self.eps = kwargs.get("eps", 1e-8)
        self.sim_block_size = kwargs.get("sim_block_size", 1024)
        self.knn_cache_enabled = kwargs.get("knn_cache_enabled", True)
        self.knn_cache_dir = kwargs.get("knn_cache_dir", "./knn_cache")
        self.knn_search_dtype = kwargs.get("knn_search_dtype", "auto")
        self.ann_large_datasets = kwargs.get(
            "ann_large_datasets", ("tsocial", "dgraphfin")
        )
        self.ann_backend = kwargs.get("ann_backend", "faiss_ivfpq")
        self.ann_nlist = kwargs.get("ann_nlist", 4096)
        self.ann_nprobe = kwargs.get("ann_nprobe", 16)
        self.ann_pq_m = kwargs.get("ann_pq_m", 16)
        self.ann_train_size = kwargs.get("ann_train_size", 262_144)
        self.ann_query_batch_size = kwargs.get("ann_query_batch_size", 4_096)
        self.ann_add_batch_size = kwargs.get("ann_add_batch_size", 262_144)
        self.ann_rerank_factor = kwargs.get("ann_rerank_factor", 32)
        self.ann_max_rerank_candidates = kwargs.get(
            "ann_max_rerank_candidates", 256
        )
        self.ann_score_batch_size = kwargs.get("ann_score_batch_size", 2_048)
        self.ann_seed = kwargs.get("ann_seed", 0)

        # Backward-compatible aliases (for old config/checkpoint consumers)
        self.lambda_ortho = self.lambda_H
        self.lambda_min_usage = self.lambda_bal
        self.lambda_diversity = self.lambda_E
        self.min_cluster_ratio = self.gamma

        self.ego_clusters = nn.ModuleList(
            [
                EgoCluster(
                    self.embed_dim,
                    num_clusters,
                    knn_k,
                    self.tau_s,
                    self.tau_c,
                    self.cluster_init_gain,
                    self.tau_e,
                    self.beta,
                    self.lambda_H,
                    self.lambda_bal,
                    self.lambda_E,
                    self.lambda_usage_entropy,
                    self.assignment_entropy_lower,
                    self.assignment_entropy_upper,
                    self.usage_entropy_lower,
                    self.usage_entropy_upper,
                    self.gamma,
                    self.eps,
                    self.sim_block_size,
                    self.knn_cache_enabled,
                    self.knn_cache_dir,
                    self.knn_search_dtype,
                    self.ann_large_datasets,
                    self.ann_backend,
                    self.ann_nlist,
                    self.ann_nprobe,
                    self.ann_pq_m,
                    self.ann_train_size,
                    self.ann_query_batch_size,
                    self.ann_add_batch_size,
                    self.ann_rerank_factor,
                    self.ann_max_rerank_candidates,
                    self.ann_score_batch_size,
                    self.ann_seed,
                )
            ]
        )

        self._view_embeds: list = []
        self._view_embeds_init: list = []
        self._view_cache_keys: list = []

    def _compute_residual_embed(self, hop_list):
        if len(hop_list) - 1 != self.num_hops:
            raise ValueError(
                f"hop_list 长度应为 num_hops+1={self.num_hops + 1}，"
                f"实际收到 {len(hop_list)}"
            )

        processed = list(hop_list)
        for i, layer in enumerate(self.layers):
            if i != 0:
                processed = [self.dropout(x) for x in processed]
            processed = [layer(x) for x in processed]
            if i != len(self.layers) - 1:
                processed = [self.act(x) for x in processed]

        first_element = processed[0]
        residual_list = [h_i - first_element for h_i in processed[1:]]
        return torch.hstack(residual_list)

    def _compute_initial_residual_embed(self, hop_list):
        """Compute initial residual embedding before MLP for KNN candidate selection."""
        first_element = hop_list[0]
        residual_list = [h_i - first_element for h_i in hop_list[1:]]
        return torch.hstack(residual_list)

    def _ensure_ego_clusters(self, num_views):
        if num_views != 1:
            raise ValueError(
                f"当前实现为单视图模式，仅支持 num_views=1，收到 {num_views}。"
            )

    def forward(self, h):
        x_list = h.x_list
        if x_list is None:
            raise RuntimeError(
                "h.x_list 为 None，请先调用 dataset.propagated(num_hops) 完成特征传播。"
            )

        is_multiview_data = isinstance(x_list[0], list)
        if is_multiview_data:
            raise ValueError("当前 recap 主线已切换为单视图模式，不支持多视图输入。")

        self._ensure_ego_clusters(1)
        residual_embed = self._compute_residual_embed(x_list)
        initial_residual_embed = self._compute_initial_residual_embed(x_list)
        self._view_embeds = [residual_embed]
        self._view_embeds_init = [initial_residual_embed]
        dataset_name = getattr(h, "dataset_name", None)
        feature_version = getattr(h, "feature_alignment_version", None)
        feature_dims = getattr(h, "feature_dims", None)
        adjacency_version = getattr(h, "adjacency_version", None)

        if dataset_name is None:
            cache_key = ("runtime", id(h), self.num_hops)
        else:
            cache_key = (
                "dataset",
                str(dataset_name),
                str(feature_version),
                str(adjacency_version),
                int(feature_dims) if feature_dims is not None else None,
                self.num_hops,
            )
        self._view_cache_keys = [cache_key]

        return residual_embed

    def get_cluster_loss(self) -> torch.Tensor:
        if not self._view_embeds:
            raise RuntimeError("请先调用 forward() 以生成视图嵌入缓存。")

        actual_num_views = len(self._view_embeds)
        if len(self._view_embeds_init) != actual_num_views:
            raise RuntimeError("初始残差缓存缺失，请重新执行 forward()。")
        if len(self._view_cache_keys) != actual_num_views:
            raise RuntimeError("KNN缓存键缺失，请重新执行 forward()。")
        return sum(
            cluster.compute_losses(E_v, E0_v, cache_key=cache_key)
            for E_v, E0_v, cache_key, cluster in zip(
                self._view_embeds,
                self._view_embeds_init,
                self._view_cache_keys,
                self.ego_clusters[:actual_num_views],
            )
        )

    @torch.no_grad()
    def get_ego_scores(self) -> torch.Tensor:
        if not self._view_embeds:
            raise RuntimeError("请先调用 forward() 以生成视图嵌入缓存。")

        return self.ego_clusters[0].compute_scores(
            self._view_embeds[0],
            E_init=self._view_embeds_init[0],
            cache_key=self._view_cache_keys[0],
        )

    @torch.no_grad()
    def get_ego_score_components(self) -> dict:
        """Return decomposed score components after cross-view aggregation."""
        if not self._view_embeds:
            raise RuntimeError("请先调用 forward() 以生成视图嵌入缓存。")

        out = self.ego_clusters[0].compute_score_components(
            self._view_embeds[0],
            E_init=self._view_embeds_init[0],
            cache_key=self._view_cache_keys[0],
        )
        out["selected_view"] = torch.zeros_like(out["total"], dtype=torch.long)
        return out

    @torch.no_grad()
    def get_ego_diagnostics(self) -> dict:
        if not self._view_embeds:
            raise RuntimeError("请先调用 forward() 以生成视图嵌入缓存。")

        E = self._view_embeds[0]
        cluster = self.ego_clusters[0]
        H = cluster.cluster(E)
        components = cluster.compute_score_components(
            E,
            E_init=self._view_embeds_init[0],
            cache_key=self._view_cache_keys[0],
        )

        n, c = H.shape
        log_c = torch.log(H.new_tensor(max(c, 2)))
        entropy = -(H * torch.log(H + cluster.eps)).sum(dim=1)
        max_prob = H.max(dim=1).values
        pi = H.mean(dim=0)
        pi_entropy = -(pi * torch.log(pi + cluster.eps)).sum()
        residual_std = E.std(dim=0, unbiased=False)
        l_var = torch.relu(cluster.gamma - residual_std).mean()
        l_var_active_ratio = (residual_std < cluster.gamma).float().mean()

        adhesion_std = components["adhesion"].std(unbiased=False)
        scale_std = components["scale"].std(unbiased=False)

        return {
            "assignment_entropy_norm": entropy.mean() / log_c,
            "assignment_max_prob": max_prob.mean(),
            "pi_entropy_norm": pi_entropy / log_c,
            "pi_std": pi.std(unbiased=False),
            "effective_soft_communities": torch.exp(pi_entropy),
            "l_var": l_var,
            "l_var_active_ratio": l_var_active_ratio,
            "adhesion_std": adhesion_std,
            "scale_std": scale_std,
            "scale_std_over_adhesion_std": scale_std / adhesion_std.clamp(min=cluster.eps),
        }
