"""
通用工具函数模块
包含数据加载、评估、模型训练等核心功能
"""
import random
import torch
import os
import pickle as pkl
from torch_geometric.data import Data
import json
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Optional, Any

# 多视图数据集名称集合，Dataset 初始化时据此走不同分支
MULTIVIEW_DATASETS = {'dblp', 'imdb', 'cert'}
FEATURE_ALIGNMENT_VERSION = 'robust_pca_post_zscore_v1'
LARGE_FEATURE_ALIGNMENT_VERSION = 'robust_sampled_pca_post_zscore_v1'
ADJACENCY_VERSION = 'sym_norm_with_self_loops_v1'
LARGE_DATASET_NAMES = {'tfinance', 'tsocial', 'dgraphfin'}


def _cache_version(npz_file) -> str:
    if 'alignment_version' not in npz_file:
        return ''
    version = npz_file['alignment_version']
    return str(version.item() if hasattr(version, 'item') else version)

'''
基础工具
'''
def test_eval(labels, probs):
    score = {}
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    score['AUROC'] = roc_auc_score(labels, probs)
    score['AUPRC'] = average_precision_score(labels, probs)
    return score


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(
        indices, values, shape, dtype=torch.float32
    )


def feat_alignment(X, dims):
    """Align features following RECAP Eq. 1-3.

    Pipeline:
      robust coordinate normalization by median/IQR,
      PCA projection to the shared dimension,
      coordinate-wise z-score post-normalization.
    """
    if torch.is_tensor(X):
        X = X.cpu().numpy()

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f'Feature matrix must be 2-D, got shape {X.shape}')

    eps = 1e-8

    # Eq. 1: robust pre-normalization.
    median = np.median(X, axis=0, keepdims=True)
    q75 = np.percentile(X, 75, axis=0, keepdims=True)
    q25 = np.percentile(X, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    X_norm = (X - median) / (iqr + eps)

    # Eq. 2: PCA projection. PCA cannot expand rank; pad only when a dataset
    # has fewer available components than the project-wide shared dimension.
    max_components = min(X_norm.shape[0], X_norm.shape[1], dims)
    if max_components <= 0:
        raise ValueError(f'Cannot run PCA for feature matrix with shape {X.shape}')

    pca = PCA(n_components=max_components, random_state=0)
    X_projected = pca.fit_transform(X_norm)

    # Eq. 3: coordinate-wise post-normalization.
    mean = X_projected.mean(axis=0, keepdims=True)
    std = X_projected.std(axis=0, keepdims=True)
    X_aligned = (X_projected - mean) / (std + eps)

    if max_components < dims:
        pad_width = dims - max_components
        X_aligned = np.pad(X_aligned, ((0, 0), (0, pad_width)), mode='constant')

    return torch.FloatTensor(X_aligned.astype(np.float32))


def feat_alignment_large_to_file(
    features,
    dims,
    output_path,
    sample_size=200_000,
    chunk_size=100_000,
):
    """Memory-bounded feature alignment for million-node graphs."""
    features = np.asarray(features)
    num_nodes, num_features = features.shape
    sample_count = min(num_nodes, sample_size)
    sample_idx = np.linspace(
        0, num_nodes - 1, num=sample_count, dtype=np.int64
    )
    sample = np.asarray(features[sample_idx], dtype=np.float32)
    eps = 1e-8
    median = np.median(sample, axis=0, keepdims=True)
    q75 = np.percentile(sample, 75, axis=0, keepdims=True)
    q25 = np.percentile(sample, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    sample = (sample - median) / (iqr + eps)
    components = min(sample_count, num_features, dims)
    pca = PCA(n_components=components, random_state=0)
    pca.fit(sample)

    aligned = np.lib.format.open_memmap(
        output_path, mode='w+', dtype=np.float32, shape=(num_nodes, dims)
    )
    aligned[:] = 0
    feature_sum = np.zeros(components, dtype=np.float64)
    feature_sq_sum = np.zeros(components, dtype=np.float64)
    for start in range(0, num_nodes, chunk_size):
        stop = min(start + chunk_size, num_nodes)
        chunk = np.asarray(features[start:stop], dtype=np.float32)
        projected = pca.transform((chunk - median) / (iqr + eps))
        projected = projected.astype(np.float32, copy=False)
        aligned[start:stop, :components] = projected
        feature_sum += projected.sum(axis=0, dtype=np.float64)
        feature_sq_sum += np.square(
            projected, dtype=np.float64
        ).sum(axis=0, dtype=np.float64)
    mean = feature_sum / num_nodes
    variance = np.maximum(feature_sq_sum / num_nodes - mean ** 2, 0)
    std = np.sqrt(variance)
    for start in range(0, num_nodes, chunk_size):
        stop = min(start + chunk_size, num_nodes)
        aligned[start:stop, :components] = (
            aligned[start:stop, :components] - mean
        ) / (std + eps)
    aligned.flush()
    return aligned


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.toarray()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def read_json(model, json_dir):
    """读取JSON配置文件"""
    filename = f"{json_dir}/{model}_10.json"
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {filename}: {e}")
                return None
    else:
        print(f"JSON file {filename} not found.")
        return None


def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

'''
数据准备
'''
def prepare_datasets(dims: int,
                    train_datasets: List[str],
                    test_datasets: List[str],
                    num_hops: int) -> tuple:
    """
    准备训练和测试数据集
    
    Args:
        dims: 特征维度
        train_datasets: 训练数据集名称列表
        test_datasets: 测试数据集名称列表
        num_hops: 传播跳数
        
    Returns:
        (data_train, data_test) 元组
    """
    print(f'加载 {len(train_datasets)} 个训练数据集: {train_datasets}')
    print(f'加载 {len(test_datasets)} 个测试数据集: {test_datasets}')
    
    # 加载数据集
    data_train = [Dataset(dims, name) for name in train_datasets]
    data_test = [Dataset(dims, name) for name in test_datasets]
    
    # 特征传播
    print(f'进行 {num_hops} 跳特征传播...')
    for tr_data in data_train:
        tr_data.propagated(num_hops)
    for te_data in data_test:
        te_data.propagated(num_hops)
    
    return data_train, data_test


class Dataset:
    def __init__(self, dims, name='cora', prefix='./dataset/'):
        # initiation
        self.graph = None
        self.x_list = None
        self.name = name
        self.is_multiview = name in MULTIVIEW_DATASETS
        self.num_views = None  # 将在初始化时设置

        # 多视图数据集走独立分支，原有单视图逻辑保持不变
        if self.is_multiview:
            self._init_multiview(dims, name, prefix)
            return

        # 单视图数据集处理
        preprocess_filename = f'{prefix}{name}_{dims}.npz'
        data = None
        feat = None
        large_dir = os.path.join(prefix, name)
        if name.lower() in LARGE_DATASET_NAMES and os.path.isdir(large_dir):
            self._init_large_singleview(dims, name, large_dir)
            return
        if os.path.exists(preprocess_filename):
            with np.load(preprocess_filename, allow_pickle=True) as f:
                if _cache_version(f) == FEATURE_ALIGNMENT_VERSION:
                    data = f['data'].item()
                    feat = torch.FloatTensor(f['feat'])  # 统一为 Tensor，与非缓存路径一致
        if data is None or feat is None:
            data = sio.loadmat(f"{prefix + name}.mat")
            adj = data['Network']
            feat = data['Attributes']
            if name in ['Amazon', 'YelpChi', 'tfinance']:
                feat = sp.lil_matrix(feat)
                feat = preprocess_features(feat)
            else:
                feat = sp.lil_matrix(feat).toarray()
            feat = torch.FloatTensor(feat)
            feat = feat_alignment(feat, dims)
            np.savez(
                preprocess_filename,
                data=data,
                feat=feat.numpy(),
                alignment_version=FEATURE_ALIGNMENT_VERSION,
            )

        adj = data['Network'] if 'Network' in data else data['A']
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        label = data['Label'] if ('Label' in data) else data['gnd']

        self.label = label
        self.adj_norm = adj_norm
        self.feat = feat
        self.num_views = 1  # 单视图数据集
        
        ano_labels = torch.tensor(np.squeeze(np.array(self.label)), dtype=torch.float)
        # Create a PyTorch Geometric Data object
        data = Data(x=torch.tensor(self.feat, dtype=torch.float),
                    x_list=self.x_list,
                    adj=self.adj_norm,
                    num_views=1,  # 单视图
                    ano_labels=ano_labels,
                    dataset_name=name,
                    feature_alignment_version=FEATURE_ALIGNMENT_VERSION,
                    feature_dims=dims,
                    adjacency_version=ADJACENCY_VERSION,
                    )
        self.graph = data

    def _init_large_singleview(self, dims, name, dataset_dir):
        """Load the disk-friendly bundle produced by the large-data tools."""
        metadata_path = os.path.join(dataset_dir, 'metadata.json')
        adjacency_path = os.path.join(dataset_dir, 'adjacency.npz')
        features_path = os.path.join(dataset_dir, 'features.npy')
        labels_path = os.path.join(dataset_dir, 'labels.npy')
        required = [metadata_path, adjacency_path, features_path, labels_path]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f'Large-dataset bundle for {name} is incomplete: {missing}'
            )

        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        adj = sp.load_npz(adjacency_path).tocsr()
        feat_raw = np.load(features_path, mmap_mode='r')
        label = np.load(labels_path, mmap_mode='r')
        if adj.shape[0] != feat_raw.shape[0] or label.shape[0] != feat_raw.shape[0]:
            raise ValueError(
                f'Inconsistent {name} bundle: adjacency={adj.shape}, '
                f'features={feat_raw.shape}, labels={label.shape}'
            )

        aligned_path = os.path.join(
            dataset_dir,
            f'features_aligned_{dims}_{LARGE_FEATURE_ALIGNMENT_VERSION}.npy',
        )
        if os.path.exists(aligned_path):
            feat_array = np.load(aligned_path, mmap_mode='r')
        else:
            feat_array = feat_alignment_large_to_file(
                feat_raw, dims, aligned_path
            )
        feat = torch.from_numpy(np.asarray(feat_array))

        adj_norm = sparse_mx_to_torch_sparse_tensor(
            normalize_adj(adj + sp.eye(adj.shape[0], dtype=np.float32))
        ).coalesce()
        ano_labels = torch.from_numpy(
            np.asarray(label, dtype=np.float32).reshape(-1).copy()
        )
        evaluation_mask_path = os.path.join(dataset_dir, 'evaluation_mask.npy')
        evaluation_mask = (
            torch.from_numpy(
                np.asarray(
                    np.load(evaluation_mask_path, mmap_mode='r'),
                    dtype=np.bool_,
                ).copy()
            )
            if os.path.exists(evaluation_mask_path)
            else torch.ones(ano_labels.shape[0], dtype=torch.bool)
        )
        self.label = label
        self.adj_norm = adj_norm
        self.feat = feat
        self.num_views = 1
        self.graph = Data(
            x=feat,
            x_list=None,
            adj=adj_norm,
            num_views=1,
            ano_labels=ano_labels,
            evaluation_mask=evaluation_mask,
            dataset_name=name,
            feature_alignment_version=LARGE_FEATURE_ALIGNMENT_VERSION,
            feature_dims=dims,
            adjacency_version=ADJACENCY_VERSION,
            source_format=metadata.get('source_format', 'canonical-large-v1'),
        )
        print(
            f'[{name}] canonical bundle loaded: nodes={feat.shape[0]}, '
            f'edges={adj.nnz}, dims={feat.shape[1]}, '
            f'anomaly_ratio={ano_labels.mean().item():.4f}'
        )

    def _init_multiview(self, dims, name, prefix):
        """初始化多视图数据集（dblp / imdb / cert）。

        加载所有视图的邻接矩阵和节点特征，对齐到统一维度后构建
        graph 对象。graph 额外携带 adj_list 字段保存所有视图的
        归一化邻接张量，num_views 字段记录视图数量，便于后续多视图模型使用。
        """
        if name == 'cert':
            adj_list_sp, feat_raw, label = self._load_cert(prefix)
        else:
            adj_list_sp, feat_raw, label = self._load_multiview_mat(name, prefix)

        # 特征对齐（PCA 降/升维），并缓存到 npz
        preprocess_filename = f'{prefix}{name}_{dims}.npz'
        feat = None
        if os.path.exists(preprocess_filename):
            with np.load(preprocess_filename, allow_pickle=True) as f:
                if _cache_version(f) == FEATURE_ALIGNMENT_VERSION:
                    feat = torch.FloatTensor(f['feat'])
                    print(f'从缓存加载 {name} 特征: {preprocess_filename}')
        if feat is None:
            feat_tensor = torch.FloatTensor(np.array(feat_raw, dtype=np.float32))
            feat = feat_alignment(feat_tensor, dims)
            np.savez(
                preprocess_filename,
                feat=feat.numpy(),
                alignment_version=FEATURE_ALIGNMENT_VERSION,
            )
            print(f'特征对齐完成并缓存到: {preprocess_filename}')

        # 归一化所有视图的邻接矩阵
        def _norm(adj_sp):
            return sparse_mx_to_torch_sparse_tensor(
                normalize_adj(adj_sp + sp.eye(adj_sp.shape[0]))
            )

        adj_norm_list = [_norm(adj_sp) for adj_sp in adj_list_sp]
        num_views = len(adj_norm_list)

        self.feat = feat
        self.label = label
        self.adj_norm = adj_norm_list[0]  # 兼容单视图接口，使用第一个视图
        self.adj_norm_list = adj_norm_list
        self.num_views = num_views

        ano_labels = torch.tensor(
            np.squeeze(np.array(label)), dtype=torch.float
        )
        self.graph = Data(
            x=feat,
            x_list=None,  # 将由 propagated() 填充
            adj=adj_norm_list,  # 所有视图的归一化邻接列表，注意后续当前还不兼容多视图情况
            num_views=num_views,  # 视图数量
            ano_labels=ano_labels,
            dataset_name=name,
            feature_alignment_version=FEATURE_ALIGNMENT_VERSION,
            feature_dims=dims,
            adjacency_version=ADJACENCY_VERSION,
        )
        print(
            f'[{name}] 节点数={feat.shape[0]}, 特征维度={feat.shape[1]}, '
            f'视图数={num_views}, 异常比例={ano_labels.mean().item():.3f}'
        )

    def _load_multiview_mat(self, name, prefix):
        """加载 dblp / imdb 的 .mat 多视图文件。

        mat 文件字段：
          feature  (N, F)   节点特征
          label    (N, 1)   异常标签（1=异常，0=正常）
          adj1     (N, N)   视图一邻接矩阵
          adj2     (N, N)   视图二邻接矩阵
          ... 可能有更多视图 adj3, adj4 等
        
        返回：
          adj_list: 所有视图邻接矩阵的列表
          feat: 特征矩阵
          label: 标签数组
        """
        filename_map = {
            'dblp': 'dblp_anomaly',
            'imdb': 'imdb5k_anomaly',
        }
        data = sio.loadmat(f'{prefix}{filename_map[name]}.mat')
        feat  = data['feature'].astype(np.float32)
        label = data['label'].reshape(-1,)
        
        # 自动检测所有视图（adj1, adj2, adj3, ...）
        adj_list = []
        i = 1
        while f'adj{i}' in data:
            adj = sp.csr_matrix(data[f'adj{i}'])
            adj_list.append(adj)
            i += 1
        
        # 如果没有找到任何 adjX 字段，尝试其他可能的命名
        if len(adj_list) == 0:
            raise ValueError(f"未找到邻接矩阵字段（adj1, adj2, ...）在 {name} 数据集中")
        
        adj_shapes = [adj.shape for adj in adj_list]
        print(
            f'[{name}] 加载完成: 节点={feat.shape[0]}, '
            f'特征={feat.shape[1]}, 视图数={len(adj_list)}, '
            f'邻接矩阵形状={adj_shapes}'
        )
        return adj_list, feat, label

    def _load_cert(self, prefix, d=100):
        """加载 CERT 内部威胁数据集（email + logon 双视图）。

        视图一（email）：v1['graph'] 已为用户-用户方阵，直接使用。
        视图二（logon）：v2['graph'] 为 PC×用户 二部图，筛选重叠节点
                        后通过 v.T @ v 投影成用户-用户方阵。
        特征：使用预先计算好的 deepwalk 嵌入（emb_{d} 文件）。
        标签：遍历 logon 用户字典，对照恶意用户列表构建。
        
        返回：
          adj_list: 所有视图邻接矩阵的列表 [adj1, adj2]
          feat: 特征矩阵
          label: 标签数组
        """
        cert_dir = os.path.join(prefix, 'CERT')

        with open(os.path.join(cert_dir, 'email.pkl'), 'rb') as f:
            v1 = pkl.load(f)
        with open(os.path.join(cert_dir, 'logon.pkl'), 'rb') as f:
            v2 = pkl.load(f)
        with open(os.path.join(cert_dir, 'label.pkl'), 'rb') as f:
            malicious_user = pkl.load(f)['label']

        # 筛选 logon 与 email 视图都有记录的 PC（行）
        overlapped_pc_idx = [
            v1['pc_dict'][item]
            for item in v2['pc_dict']
            if item in v1['pc_dict']
        ]
        v2['graph']  = v2['graph'][overlapped_pc_idx, :]
        v2['weight'] = v2['weight'][overlapped_pc_idx, :]

        # 筛选重叠用户（列），同步构造节点标签
        overlapped_user_idx = []
        label = []
        for item in v2['user_dict']:
            if item in v1['user_dict']:
                overlapped_user_idx.append(v1['user_dict'][item])
            label.append(1 if item in malicious_user else 0)
        v2['graph']  = v2['graph'][:, overlapped_user_idx]
        v2['weight'] = v2['weight'][:, overlapped_user_idx]

        # 读取已有的 deepwalk 嵌入特征
        n_users_email  = v1['weight'].shape[0]
        n_users_logon  = v2['weight'].shape[1]   # 筛选后的用户列数
        v1_feature = np.zeros((n_users_email, d), dtype=np.float32)
        v2_feature = np.zeros((n_users_logon, d), dtype=np.float32)

        # 注意：文件名与视图对象看似对调，与原始 utils2.py 保持一致
        # email 视图(v1)使用 logon_edge_list_emb，logon 视图(v2)使用 email_edge_list_emb
        with open(os.path.join(cert_dir, f'logon_edge_list_emb_{d}'), 'r') as f:
            next(f)
            for line in f:
                parts = list(map(float, line.split()))
                node_id = int(parts[0])
                if node_id >= 1000:
                    idx = node_id - 1000
                    if idx < n_users_email:
                        v1_feature[idx] = parts[1:]

        with open(os.path.join(cert_dir, f'email_edge_list_emb_{d}'), 'r') as f:
            next(f)
            for line in f:
                parts = list(map(float, line.split()))
                node_id = int(parts[0])
                if node_id >= 1000:
                    idx = node_id - 1000
                    if idx < n_users_logon:
                        v2_feature[idx] = parts[1:]

        # 视图一邻接：email 用户-用户方阵，裁剪到与 logon 视图相同的重叠用户子集
        # overlapped_user_idx 中存的是 v1 侧的用户索引
        adj1_full = sp.csr_matrix(v1['graph'])
        adj1 = adj1_full[overlapped_user_idx, :][:, overlapped_user_idx]

        # 视图二邻接：logon 二部图 → 用户-用户方阵（v.T @ v）
        v2_sp = sp.csr_matrix(v2['graph'])
        if v2_sp.shape[0] != v2_sp.shape[1]:
            adj2 = v2_sp.T.dot(v2_sp)   # (n_users × n_users)
        else:
            adj2 = v2_sp

        label = np.array(label, dtype=np.float32)
        
        # 将所有视图放入列表
        adj_list = [adj1, adj2]
        
        print(
            f'[cert] 加载完成: email用户={n_users_email}, '
            f'logon用户={n_users_logon}, 视图数={len(adj_list)}, '
            f'邻接矩阵形状={[adj.shape for adj in adj_list]}, 标签数={len(label)}'
        )
        # 特征与 logon 视图对齐（与原始 utils2.py 保持一致）
        return adj_list, v2_feature, label

    def propagated(self, k, device=None):
        """特征传播（支持单视图和多视图）
        
        Args:
            k: 传播跳数
            device: 计算设备 ('cuda', 'cpu' 或 torch.device)，默认使用 CUDA（如果可用）
            
        对于单视图数据集:
            x_list = [X, AX, A²X, ..., A^kX]
            
        对于多视图数据集:
            x_list = [
                [X, A1·X, A1²·X, ..., A1^k·X],  # 视图1的传播
                [X, A2·X, A2²·X, ..., A2^k·X],  # 视图2的传播
                ...
            ]
        """
        # 自动选择设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        x = torch.FloatTensor(self.feat).to(device)
        
        if self.is_multiview:
            # 多视图：每个视图分别传播
            x_list_all_views = []
            for adj_norm in self.adj_norm_list:
                h_list = [x]
                for _ in range(k):
                    h_list.append(torch.spmm(adj_norm.to(device), h_list[-1]))
                x_list_all_views.append(h_list)
            
            # 保存为嵌套列表：[[view1的k+1层], [view2的k+1层], ...]
            self.graph.x_list = x_list_all_views
            
            print(f'多视图传播完成: {len(x_list_all_views)} 个视图, '
                  f'每个视图 {len(x_list_all_views[0])} 层特征 (设备: {device})')
        else:
            # 单视图：使用主邻接矩阵传播
            h_list = [x]
            for _ in range(k):
                h_list.append(torch.spmm(self.adj_norm.to(device), h_list[-1]))
            self.graph.x_list = h_list
