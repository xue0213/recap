"""
recap 异常检测器模块
包含核心的检测器类，封装训练和评估逻辑
"""
import torch
from typing import Dict, List, Optional, Any
from utils import test_eval
from model import recap

class recapDetector:
    """recap异常检测器 - 封装训练和评估逻辑
    
    这是项目的核心类，负责：
    1. 模型的训练
    2. 模型的评估
    3. 训练过程的管理

    """
    
    def __init__(self, train_config, model_config, data):
        """
        初始化recap检测器
        
        Args:
            train_config: 训练配置（dict或TrainConfig对象）
                必需字段: device, epochs
            model_config: 模型配置（dict或ModelConfig对象）
                必需字段: dims, num_layers, lr, weight_decay
            data: 数据字典，必须包含'train'和'test'键
                格式: {'train': List[Dataset], 'test': List[Dataset]}
        """
        
        # 兼容dict和dataclass两种类型
        if hasattr(model_config, 'to_dict'):
            self.model_config = model_config.to_dict()
        else:
            self.model_config = model_config
            
        if hasattr(train_config, 'to_dict'):
            self.train_config = train_config.to_dict()
        else:
            self.train_config = train_config
            
        self.data = data
        
        # 主线实现固定为单视图模式。
        self.model_config['num_views'] = 1
        print('🔧 模型初始化视图数: 1 (单视图模式)')
        
        self.model = recap(**self.model_config).to(self.train_config['device'])
        self.optimizer = None
    
    def _init_optimizer(self):
        """初始化优化器（延迟初始化）"""
        if self.optimizer is None:
            lr = self.model_config['lr']
            weight_decay = self.model_config['weight_decay']
            cluster_lr_multiplier = float(self.model_config.get('cluster_lr_multiplier', 1.0))

            if cluster_lr_multiplier == 1.0:
                params = self.model.parameters()
            else:
                cluster_param_ids = {
                    id(param) for param in self.model.ego_clusters.parameters()
                }
                base_params = [
                    param for param in self.model.parameters()
                    if id(param) not in cluster_param_ids
                ]
                cluster_params = list(self.model.ego_clusters.parameters())
                params = [
                    {'params': base_params, 'lr': lr, 'weight_decay': weight_decay},
                    {
                        'params': cluster_params,
                        'lr': lr * cluster_lr_multiplier,
                        'weight_decay': 0.0,
                    },
                ]
                print(
                    f'🔧 Assignment layer LR multiplier: {cluster_lr_multiplier:g} '
                    f'(base_lr={lr:g}, assignment_lr={lr * cluster_lr_multiplier:g})'
                )

            self.optimizer = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay
            )

    @torch.no_grad()
    def _collect_train_diagnostics(self) -> Dict[str, float]:
        self.model.eval()
        stats: Dict[str, list] = {}

        for train_data in self.data['train']:
            train_graph = train_data.graph.to(self.train_config['device'])
            self.model(train_graph)
            diag = self.model.get_ego_diagnostics()
            for key, value in diag.items():
                stats.setdefault(key, []).append(float(value.detach().cpu().item()))

        self.model.train()
        return {
            key: sum(values) / len(values)
            for key, values in stats.items()
            if values
        }

    def _maybe_log_diagnostics(self, epoch: int):
        if not self.train_config.get('log_diagnostics', True):
            return

        interval = int(self.train_config.get('diagnostics_interval', 10))
        if interval <= 0:
            return
        if epoch != 0 and (epoch + 1) % interval != 0:
            return

        diag = self._collect_train_diagnostics()
        if not diag:
            return

        print(
            '  Diagnostics: '
            f'H_ent_norm={diag["assignment_entropy_norm"]:.4f}, '
            f'H_maxp={diag["assignment_max_prob"]:.4f}, '
            f'pi_ent_norm={diag["pi_entropy_norm"]:.4f}, '
            f'pi_std={diag["pi_std"]:.6f}, '
            f'effC={diag["effective_soft_communities"]:.2f}, '
            f'Lvar={diag["l_var"]:.6f}, '
            f'Lvar_active={diag["l_var_active_ratio"]:.4f}, '
            f'scale/adh_std={diag["scale_std_over_adhesion_std"]:.6g}'
        )
    
    def train_epoch(self, epoch: int) -> float:
        """
        训练单个epoch

        使用 RECAP 损失：
            L = L_con + λ_H * L_H + λ_E * L_var
        
        Args:
            epoch: 当前epoch编号（从0开始）
            
        Returns:
            float: 该epoch的平均损失
        """
        self._init_optimizer()
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for didx, train_data in enumerate(self.data['train']):
            train_graph = train_data.graph.to(self.train_config['device'])

            # forward() 计算并缓存单视图 residual 表示
            self.model(train_graph)

            # RECAP 训练损失
            loss = self.model.get_cluster_loss()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{self.train_config["epochs"]}], '
                  f'Average Loss: {avg_loss:.6f}')
        self._maybe_log_diagnostics(epoch)
        
        return avg_loss
    
    def train(self, verbose: bool = True, early_stop: Optional[bool] = None,
              patience: Optional[int] = None) -> Dict[str, Any]:
        """
        完整训练流程（支持early stopping）
        
        Args:
            verbose: 是否打印详细训练信息
            early_stop: 是否启用early stopping
            patience: early stopping耐心值（连续多少轮loss不降则停止）
            
        Returns:
            Dict[str, Any]: 训练历史字典
                - losses: List[float] - 每个epoch的损失
                - epochs: int - 实际训练epoch数
                - stopped_early: bool - 是否early stop
        """
        if early_stop is None:
            early_stop = bool(self.train_config.get('early_stop', False))
        if patience is None:
            patience = int(self.train_config.get('patience', 30))

        if verbose:
            print(f'开始训练，共 {self.train_config["epochs"]} 个epochs...')
            if early_stop:
                print(f'启用early stopping (patience={patience})')
            else:
                print('关闭early stopping，训练将跑满设定epochs')
        
        train_history = {'losses': [], 'epochs': 0, 'stopped_early': False}
        best_loss = float('inf')
        patience_counter = 0
        
        for e in range(self.train_config['epochs']):
            loss = self.train_epoch(e)
            train_history['losses'].append(loss)
            
            # Early stopping检查
            if early_stop:
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {e+1} (best loss: {best_loss:.6f})')
                    train_history['stopped_early'] = True
                    train_history['epochs'] = e + 1
                    break
        
        if not train_history['stopped_early']:
            train_history['epochs'] = self.train_config['epochs']
        
        if verbose:
            print(f'训练完成！实际训练 {train_history["epochs"]} 个epochs')
        
        return train_history
    
    def evaluate(self,
                data_list: Optional[List] = None,
                dataset_names: Optional[List[str]] = None,
                verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        评估模型性能

        使用 KNN ego graph 社区上下文不一致性打分：
            s_i = norm(s_adhesion) + beta * norm(JS(H_i, Q_i))
        
        Args:
            data_list: 要评估的数据集列表，如果为None则使用self.data['test']
            dataset_names: 数据集名称列表，用于结果展示
            verbose: 是否打印详细评估信息
            
        Returns:
            Dict[str, Dict[str, float]]: 测试分数字典
                格式: {
                    'dataset1': {'AUROC': 0.85, 'AUPRC': 0.78},
                    'dataset2': {'AUROC': 0.82, 'AUPRC': 0.75},
                    ...
                }
        """
        # if data_list is None:
        #     data_list = self.data['test']
        
        # if dataset_names is None:
        #     dataset_names = self.train_config.get('testdsets',
        #                                           [f'dataset_{i}' for i in range(len(data_list))])
        
        test_score_list = {}
        self.model.eval()
        
        with torch.no_grad():
            for didx, test_data in enumerate(data_list):
                test_graph = test_data.graph.to(self.train_config['device'])
                labels = test_graph.ano_labels

                # forward() 计算并缓存单视图 residual 表示
                self.model(test_graph)

                # RECAP score
                all_scores = self.model.get_ego_scores()
                query_scores = all_scores
                query_labels = labels.to(self.train_config['device'])
                evaluation_mask = getattr(test_graph, 'evaluation_mask', None)
                if evaluation_mask is not None:
                    query_labels = query_labels[evaluation_mask]
                    query_scores = query_scores[evaluation_mask]
            
                test_score = test_eval(query_labels, query_scores)
                
                test_data_name = dataset_names[didx]
                test_score_list[test_data_name] = {
                    'AUROC': test_score['AUROC'],
                    'AUPRC': test_score['AUPRC'],
                }
                
                if verbose:
                    print(f'  {test_data_name}: '
                          f'AUROC={test_score["AUROC"]:.4f}, '
                          f'AUPRC={test_score["AUPRC"]:.4f}')
        
        return test_score_list
    
    def get_model(self):
        """
        获取训练好的模型
        
        Returns:
            torch.nn.Module: 训练好的recap模型
            
        Example:
            >>> model = detector.get_model()
            >>> torch.save(model.state_dict(), 'model.pth')
        """
        return self.model
    
    def save_checkpoint(self, path: str, epoch: int = None, **kwargs):
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            epoch: 当前epoch（可选）
            **kwargs: 其他要保存的信息
            
        Example:
            >>> detector.save_checkpoint('checkpoints/model.pth', epoch=100)
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'train_config': self.train_config,
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
            
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        print(f'检查点已保存到: {path}')
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """
        加载模型检查点
        
        Args:
            path: 检查点路径
            load_optimizer: 是否加载优化器状态
            
        Returns:
            dict: 检查点中的其他信息
            
        Example:
            >>> detector.load_checkpoint('checkpoints/model.pth')
        """
        checkpoint = torch.load(path, map_location=self.train_config['device'])
        
        load_result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                '检查点参数与当前模型存在非严格匹配：'
                f'missing={load_result.missing_keys}, '
                f'unexpected={load_result.unexpected_keys}'
            )
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self._init_optimizer()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f'检查点已加载: {path}')
        
        return {k: v for k, v in checkpoint.items() 
                if k not in ['model_state_dict', 'optimizer_state_dict']}
