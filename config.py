"""
配置管理模块
提供训练、模型和推理的配置类
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json
import os


@dataclass
class TrainConfig:
    """训练配置"""
    device: str = "cuda:0"
    epochs: int = 100
    trials: int = 5
    seed: int = 0
    output_dir: str = "./checkpoints"
    save_checkpoint: bool = True
    early_stop: bool = False
    patience: int = 30
    log_diagnostics: bool = True
    diagnostics_interval: int = 10

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


@dataclass
class ModelConfig:
    """模型配置（RECAP 论文对齐版）"""

    model: str = "recap"
    lr: float = 5e-5
    dropout_rate: float = 0.2
    h_feats: int = 256
    num_hops: int = 4
    weight_decay: float = 5e-5
    dims: int = 32
    num_layers: int = 2
    activation: str = "ELU"

    # Community assignment settings
    num_clusters: int = 28
    knn_k: int = 20
    cluster_lr_multiplier: float = 10.0
    tau_s: float = 0.08
    tau_c: float = 0.3
    tau_e: float = 1.0
    cluster_init_gain: Optional[float] = 1.5
    sim_block_size: int = 256
    knn_cache_enabled: bool = True
    knn_cache_dir: str = "./knn_cache"
    knn_search_dtype: str = "auto"

    # Approximate KNN is opt-in by dataset name. All other datasets keep the
    # original exact blockwise search and cache namespace.
    ann_large_datasets: tuple = ("tsocial", "dgraphfin")
    ann_backend: str = "faiss_ivfpq"
    ann_nlist: int = 4096
    ann_nprobe: int = 16
    ann_pq_m: int = 16
    ann_train_size: int = 262_144
    ann_query_batch_size: int = 4_096
    ann_add_batch_size: int = 262_144
    ann_rerank_factor: int = 32
    ann_max_rerank_candidates: int = 256
    ann_score_batch_size: int = 2_048
    ann_seed: int = 0

    # RECAP loss hyperparameters
    lambda_H: float = 0.1
    lambda_bal: float = 0.1
    lambda_E: float = 0.0
    lambda_usage_entropy: float = 0.1
    assignment_entropy_lower: Optional[float] = 0.45
    assignment_entropy_upper: Optional[float] = 0.85
    usage_entropy_lower: Optional[float] = 0.65
    usage_entropy_upper: Optional[float] = 0.9

    # RECAP inference scoring hyperparameters
    beta: float = 0.02
    gamma: float = 0.01

    # Numerical stability
    eps: float = 1e-8

    def __post_init__(self):
        pass

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        # Accept historical keys and map to RECAP keys when needed.
        cfg = dict(config_dict)
        if "lambda_H" not in cfg and "lambda_ortho" in cfg:
            cfg["lambda_H"] = cfg["lambda_ortho"]
        valid_keys = set(cls.__dataclass_fields__.keys())
        cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**cfg)

    @classmethod
    def from_json(cls, json_path):
        """从JSON文件加载配置"""
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                config_dict = json.load(f)
                return cls.from_dict(config_dict)
        return None

    def save_json(self, json_path):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class InferenceConfig:
    """推理配置"""

    checkpoint_path: str
    device: str = "cuda:0"
    batch_size: int = 1024
    output_dir: str = "./inference_results"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


class ConfigManager:
    """配置管理器"""

    @staticmethod
    def load_model_config(
        model_name: str, json_dir: str = "./params"
    ) -> Optional[ModelConfig]:
        """
        加载模型配置

        Args:
            model_name: 模型名称
            json_dir: JSON配置文件目录

        Returns:
            ModelConfig对象，如果文件不存在则返回None
        """
        filename = f"{json_dir}/{model_name}.json"
        return ModelConfig.from_json(filename)

    @staticmethod
    def get_default_model_config(dims: int = 64) -> ModelConfig:
        """
        获取默认模型配置

        Args:
            dims: 输入特征维度

        Returns:
            默认的ModelConfig对象
        """
        return ModelConfig(dims=dims)

    @staticmethod
    def save_all_configs(train_config: TrainConfig, model_config: ModelConfig, save_dir: str):
        """
        保存所有配置到指定目录

        Args:
            train_config: 训练配置
            model_config: 模型配置
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        train_config_path = os.path.join(save_dir, "train_config.json")
        with open(train_config_path, "w") as f:
            json.dump(train_config.to_dict(), f, indent=2)

        model_config_path = os.path.join(save_dir, "model_config.json")
        with open(model_config_path, "w") as f:
            json.dump(model_config.to_dict(), f, indent=2)

    @staticmethod
    def load_all_configs(load_dir: str) -> tuple:
        """
        从指定目录加载所有配置

        Args:
            load_dir: 加载目录

        Returns:
            (TrainConfig, ModelConfig) 元组
        """
        train_config_path = os.path.join(load_dir, "train_config.json")
        with open(train_config_path, "r") as f:
            train_config_dict = json.load(f)
            train_config = TrainConfig.from_dict(train_config_dict)

        model_config_path = os.path.join(load_dir, "model_config.json")
        with open(model_config_path, "r") as f:
            model_config_dict = json.load(f)
            model_config = ModelConfig.from_dict(model_config_dict)

        return train_config, model_config


def create_default_configs(
    model_name: str = "recap", json_dir: str = "./params", dims: int = 32
) -> tuple:
    """
    创建默认配置

    Args:
        model_name: 模型名称
        json_dir: JSON配置文件目录
        dims: 输入特征维度

    Returns:
        (TrainConfig, ModelConfig) 元组
    """
    train_config = TrainConfig()

    model_config = ConfigManager.load_model_config(model_name, json_dir)

    if model_config is None:
        model_config = ConfigManager.get_default_model_config(dims)
        print("使用默认模型配置")
    else:
        print("使用保存的最佳模型配置")
        model_config.dims = dims

    model_config.model = model_name

    return train_config, model_config
