"""
模型检查点管理模块
提供模型保存、加载和版本管理功能
"""
import os
import torch
from datetime import datetime
from typing import Dict, Optional, Any
from config import TrainConfig, ModelConfig, ConfigManager
from model import recap


class ModelCheckpoint:
    """模型检查点管理器"""
    
    def __init__(self, base_dir: str = './checkpoints'):
        """
        初始化检查点管理器
        
        Args:
            base_dir: 检查点保存的基础目录
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       train_config: TrainConfig,
                       model_config: ModelConfig,
                       epoch: int,
                       trial: int = 0,
                       metrics: Optional[Dict[str, Any]] = None,
                       seed: Optional[int] = None,
                       history: Optional[Dict[str, Any]] = None) -> str:
        """
        保存模型检查点
        
        Args:
            model: 训练好的模型
            train_config: 训练配置
            model_config: 模型配置
            epoch: 训练的epoch数
            trial: 试验编号
            metrics: 性能指标字典
            seed: 随机种子
            history: 训练历史（包含losses等）
            
        Returns:
            保存的检查点路径
        """
        # 创建保存目录
        model_name = model_config.model
        save_dir = os.path.join(self.base_dir, model_name, f'trial_{trial}')
        os.makedirs(save_dir, exist_ok=True)
        
        # 准备检查点数据
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config.to_dict(),
            'train_config': train_config.to_dict(),
            'epoch': epoch,
            'trial': trial,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if seed is not None:
            checkpoint['seed'] = seed
        
        if history is not None:
            checkpoint['history'] = history
        
        # 保存模型检查点
        checkpoint_path = os.path.join(save_dir, 'model.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存配置文件（便于查看）
        ConfigManager.save_all_configs(train_config, model_config, save_dir)
        
        print(f'检查点已保存到: {checkpoint_path}')
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, device: str = 'cuda:0') -> tuple:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            device: 设备
            
        Returns:
            (model, train_config, model_config, checkpoint_info) 元组
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'检查点文件不存在: {checkpoint_path}')
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 重建模型配置
        model_config = ModelConfig.from_dict(checkpoint['model_config'])
        train_config = TrainConfig.from_dict(checkpoint['train_config'])
        
        # 重建模型
        model = recap(**model_config.to_dict()).to(device)
        load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                '检查点参数与当前模型存在非严格匹配：'
                f'missing={load_result.missing_keys}, '
                f'unexpected={load_result.unexpected_keys}'
            )
        
        # 提取检查点信息
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', -1),
            'trial': checkpoint.get('trial', 0),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'metrics': checkpoint.get('metrics', {}),
            'seed': checkpoint.get('seed', None),
        }
        
        print(f'成功加载检查点: {checkpoint_path}')
        print(f'训练信息: epoch={checkpoint_info["epoch"]}, trial={checkpoint_info["trial"]}')
        
        return model, train_config, model_config, checkpoint_info
    
    def get_checkpoint_path(self, model_name: str, trial: int = 0) -> str:
        """
        获取检查点路径
        
        Args:
            model_name: 模型名称
            trial: 试验编号
            
        Returns:
            检查点文件路径
        """
        return os.path.join(self.base_dir, model_name, f'trial_{trial}', 'model.pt')
    
    def list_checkpoints(self, model_name: Optional[str] = None) -> list:
        """
        列出所有检查点
        
        Args:
            model_name: 模型名称，如果为None则列出所有模型
            
        Returns:
            检查点路径列表
        """
        checkpoints = []
        
        if model_name is not None:
            model_dir = os.path.join(self.base_dir, model_name)
            if os.path.exists(model_dir):
                for trial_dir in sorted(os.listdir(model_dir)):
                    checkpoint_path = os.path.join(model_dir, trial_dir, 'model.pt')
                    if os.path.exists(checkpoint_path):
                        checkpoints.append(checkpoint_path)
        else:
            # 列出所有模型的检查点
            if os.path.exists(self.base_dir):
                for model_name in os.listdir(self.base_dir):
                    model_dir = os.path.join(self.base_dir, model_name)
                    if os.path.isdir(model_dir):
                        for trial_dir in sorted(os.listdir(model_dir)):
                            checkpoint_path = os.path.join(model_dir, trial_dir, 'model.pt')
                            if os.path.exists(checkpoint_path):
                                checkpoints.append(checkpoint_path)
        
        return checkpoints
    
    def get_latest_checkpoint(self, model_name: str) -> Optional[str]:
        """
        获取最新的检查点
        
        Args:
            model_name: 模型名称
            
        Returns:
            最新检查点的路径，如果不存在则返回None
        """
        checkpoints = self.list_checkpoints(model_name)
        if not checkpoints:
            return None
        
        # 按修改时间排序，返回最新的
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints[0]


class CheckpointManager:
    """检查点管理器（提供更高级的功能）"""
    
    def __init__(self, base_dir: str = './checkpoints'):
        self.checkpoint = ModelCheckpoint(base_dir)
    
    def save_best_model(self,
                       model: torch.nn.Module,
                       train_config: TrainConfig,
                       model_config: ModelConfig,
                       epoch: int,
                       trial: int,
                       metrics: Dict[str, Any],
                       seed: int) -> str:
        """
        保存最佳模型
        
        Args:
            model: 模型
            train_config: 训练配置
            model_config: 模型配置
            epoch: epoch数
            trial: 试验编号
            metrics: 性能指标
            seed: 随机种子
            
        Returns:
            检查点路径
        """
        return self.checkpoint.save_checkpoint(
            model=model,
            train_config=train_config,
            model_config=model_config,
            epoch=epoch,
            trial=trial,
            metrics=metrics,
            seed=seed
        )
    
    def load_for_inference(self, checkpoint_path: str, device: str = 'cuda:0'):
        """
        加载模型用于推理
        
        Args:
            checkpoint_path: 检查点路径
            device: 设备
            
        Returns:
            (model, model_config, checkpoint_info) 元组
        """
        model, _, model_config, checkpoint_info = self.checkpoint.load_checkpoint(
            checkpoint_path, device
        )
        model.eval()  # 设置为评估模式
        return model, model_config, checkpoint_info
    
    def export_model_summary(self, checkpoint_path: str, output_path: str):
        """
        导出模型摘要信息
        
        Args:
            checkpoint_path: 检查点路径
            output_path: 输出路径
        """
        import json
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        summary = {
            'model_config': checkpoint['model_config'],
            'train_config': checkpoint['train_config'],
            'epoch': checkpoint.get('epoch', -1),
            'trial': checkpoint.get('trial', 0),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'metrics': checkpoint.get('metrics', {}),
            'seed': checkpoint.get('seed', None),
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f'模型摘要已保存到: {output_path}')


# 便捷函数
def save_model(model: torch.nn.Module,
              train_config: TrainConfig,
              model_config: ModelConfig,
              epoch: int,
              trial: int = 0,
              metrics: Optional[Dict[str, Any]] = None,
              seed: Optional[int] = None,
              save_dir: str = './checkpoints') -> str:
    """
    便捷的模型保存函数
    
    Args:
        model: 模型
        train_config: 训练配置
        model_config: 模型配置
        epoch: epoch数
        trial: 试验编号
        metrics: 性能指标
        seed: 随机种子
        save_dir: 保存目录
        
    Returns:
        检查点路径
    """
    checkpoint_manager = ModelCheckpoint(save_dir)
    return checkpoint_manager.save_checkpoint(
        model, train_config, model_config, epoch, trial, metrics, seed
    )


def load_model(checkpoint_path: str, device: str = 'cuda:0'):
    """
    便捷的模型加载函数
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        (model, train_config, model_config, checkpoint_info) 元组
    """
    checkpoint_manager = ModelCheckpoint()
    return checkpoint_manager.load_checkpoint(checkpoint_path, device)
