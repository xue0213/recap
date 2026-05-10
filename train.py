"""
训练脚本
独立的模型训练程序，训练完成后保存模型检查点
"""
import argparse
import numpy as np
import warnings
import os

from utils import set_seed, prepare_datasets
from detector import recapDetector
from config import TrainConfig, ModelConfig, create_default_configs
from model_checkpoint import ModelCheckpoint


def train_single_trial(trial: int,
                      train_config: TrainConfig,
                      model_config: ModelConfig,
                      data_train: list,
                      data_test: list,
                      test_dataset_names: list,
                      checkpoint_manager: ModelCheckpoint) -> dict:
    """
    单次试验的训练流程
    
    Args:
        trial: 试验编号
        train_config: 训练配置
        model_config: 模型配置
        data_train: 训练数据集列表
        data_test: 测试数据集列表
        test_dataset_names: 测试数据集名称列表
        checkpoint_manager: 检查点管理器
        
    Returns:
        测试结果字典
    """
    seed = trial
    set_seed(seed)
    print(f'\n{"="*70}')
    print(f'Trial {trial} 开始训练 (seed={seed})')
    print(f'{"="*70}')
    
    # 准备数据
    data = {'train': data_train, 'test': data_test}
    
    # 创建检测器
    detector = recapDetector(train_config, model_config, data)
    
    # 训练
    print('\n--- 训练阶段 ---')
    train_history = detector.train(verbose=True)
    
    # 评估
    print('\n--- 评估阶段 ---')
    test_score_list = detector.evaluate(
        data_list=data_test,
        dataset_names=test_dataset_names,
        verbose=True
    )
    
    # 保存检查点
    if train_config.save_checkpoint:
        print('\n--- 保存模型 ---')
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=detector.get_model(),
            train_config=train_config,
            model_config=model_config,
            epoch=train_config.epochs,
            trial=trial,
            metrics=test_score_list,
            seed=seed,
            history=train_history  # 保存训练历史
        )
        print(f'模型已保存到: {checkpoint_path}')
    
    return test_score_list


def aggregate_results(all_results: list, test_dataset_names: list):
    """
    聚合多次试验的结果
    
    Args:
        all_results: 所有试验的结果列表
        test_dataset_names: 测试数据集名称列表
    """
    print(f'\n{"="*70}')
    print('聚合 {} 次试验的结果'.format(len(all_results)))
    print(f'{"="*70}\n')
    
    # 收集每个数据集的分数
    auc_dict = {name: [] for name in test_dataset_names}
    pre_dict = {name: [] for name in test_dataset_names}
    
    for test_scores in all_results:
        for dataset_name, scores in test_scores.items():
            auc_dict[dataset_name].append(scores['AUROC'])
            pre_dict[dataset_name].append(scores['AUPRC'])
    
    # 计算均值和标准差
    for dataset_name in test_dataset_names:
        auc_mean = np.mean(auc_dict[dataset_name])
        auc_std = np.std(auc_dict[dataset_name])
        pre_mean = np.mean(pre_dict[dataset_name])
        pre_std = np.std(pre_dict[dataset_name])
        
        print(f'{"-"*25} {dataset_name} {"-"*25}')
        print(f'AUROC: {auc_mean:.4f} ± {auc_std:.4f}')
        print(f'AUPRC: {pre_mean:.4f} ± {pre_std:.4f}')
        print()


def main():
    
    warnings.filterwarnings("ignore")

    # ==================== 1. 参数解析 ====================
    parser = argparse.ArgumentParser(
        description='recap 异常检测模型训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 训练参数
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--trials', type=int, default=3, help='试验次数')
    train_group.add_argument('--epochs', type=int, default=200, help='训练轮数')
    train_group.add_argument('--device', type=str, default='cuda:0', help='训练设备')
    train_group.add_argument('--output-dir', type=str, default='./checkpoints', help='模型保存目录')
    train_group.add_argument('--no-save', action='store_true', help='不保存检查点')
    train_group.add_argument('--early-stop', action='store_true', help='启用early stopping')
    train_group.add_argument('--patience', type=int, default=30, help='early stopping耐心值')
    train_group.add_argument('--no-diagnostics', action='store_true', help='关闭训练期community诊断日志')
    train_group.add_argument('--diagnostics-interval', type=int, default=10, help='诊断日志间隔epoch数')
    
    # 模型参数
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--model', type=str, default='recap', help='模型名称')
    model_group.add_argument('--json-dir', type=str, default='./params', help='JSON 配置文件目录')
    model_group.add_argument('--dims', type=int, default=32, help='特征维度')
    model_group.add_argument('--lr', type=float, default=None, help='覆盖配置中的学习率')
    model_group.add_argument('--cluster-lr-multiplier', type=float, default=None,
                             help='community assignment层学习率倍率')
    model_group.add_argument('--gamma', type=float, default=None, help='覆盖L_var中的gamma阈值')
    
    # 数据集参数
    data_group = parser.add_argument_group('Dataset Parameters')
    data_group.add_argument('--train-datasets', type=str, nargs='+', default=None,
                           help='指定训练数据集（默认使用4个训练数据集）')
    data_group.add_argument('--test-datasets', type=str, nargs='+', default=None,
                           help='指定测试数据集（默认使用8个单视图数据集）')
    
    args = parser.parse_args()

    # ==================== 2. 配置创建 ====================
    train_config, model_config = create_default_configs(
        model_name=args.model,
        json_dir=args.json_dir,
        dims=args.dims
    )
    
    # 统一更新训练配置
    config_updates = {
        'device': args.device,
        'epochs': args.epochs,
        'trials': args.trials,
        'output_dir': args.output_dir,
        'save_checkpoint': not args.no_save,
        'early_stop': args.early_stop,
        'patience': args.patience,
        'log_diagnostics': not args.no_diagnostics,
        'diagnostics_interval': args.diagnostics_interval,
    }
    
    for key, value in config_updates.items():
        setattr(train_config, key, value)

    if args.lr is not None:
        model_config.lr = args.lr
    if args.cluster_lr_multiplier is not None:
        model_config.cluster_lr_multiplier = args.cluster_lr_multiplier
    if args.gamma is not None:
        model_config.gamma = args.gamma
        model_config.min_cluster_ratio = args.gamma


    print(f'\n{"="*70}')
    print('recap 异常检测模型训练')
    print(f'{"="*70}')
    
    print(f'\n📋 训练配置:')
    for key, value in train_config.to_dict().items():
        print(f'  {key}: {value}')
    
    print(f'\n🔧 模型配置:')
    for key, value in model_config.to_dict().items():
        print(f'  {key}: {value}')
    
    print(f'\n💾 保存设置:')
    print(f'  保存目录：{args.output_dir}')
    print(f'  保存检查点：{train_config.save_checkpoint}')
    print(f'{"="*70}\n')

    # ==================== 3. 数据集配置 ====================
    # 训练数据集：如果指定了--train-datasets则使用指定的，否则使用默认训练集
    if args.train_datasets:
        datasets_train = args.train_datasets
        print(f'\n📊 使用指定的训练数据集: {datasets_train}')
    else:
        datasets_train = ['pubmed', 'Flickr', 'questions', 'YelpChi']
        print(f'\n📊 使用默认训练数据集 ({len(datasets_train)}个)')
    
    # 测试数据集：如果指定了--test-datasets则使用指定的，否则使用所有数据集
    if args.test_datasets:
        datasets_test = args.test_datasets
        print(f'\n📊 使用指定的测试数据集: {datasets_test}')
    else:
        datasets_test = ['Facebook','cora', 'citeseer', 'ACM', 'BlogCatalog',
                        'weibo', 'Reddit', 'Amazon']
        print(f'\n📊 使用默认单视图测试数据集 ({len(datasets_test)}个)')
    
    # 准备数据集
    data_train, data_test = prepare_datasets(
        dims=args.dims,
        train_datasets=datasets_train,
        test_datasets=datasets_test,
        num_hops=model_config.num_hops
    )
    
    # ==================== 4. 执行训练并保存检查点 ====================
    # 创建检查点管理器
    checkpoint_manager = ModelCheckpoint(args.output_dir)
    
    # 多次试验
    all_results = []
    for trial in range(args.trials):
        test_scores = train_single_trial(
            trial=trial,
            train_config=train_config,
            model_config=model_config,
            data_train=data_train,
            data_test=data_test,
            test_dataset_names=datasets_test,
            checkpoint_manager=checkpoint_manager
        )
        all_results.append(test_scores)
    
    # 聚合结果
    aggregate_results(all_results, datasets_test)
    
    print(f'\n{"="*70}')
    print('训练完成！')
    if train_config.save_checkpoint:
        print(f'所有模型已保存到: {args.output_dir}/{args.model}/')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
