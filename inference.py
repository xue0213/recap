"""
推理脚本
使用训练好的模型进行推理和评估
"""
import argparse
import warnings
import json
import os
import numpy as np

from utils import Dataset, set_seed, test_eval
from model_checkpoint import load_model


def prepare_test_datasets(dims: int,
                         test_datasets: list,
                         num_hops: int) -> list:
    """
    准备测试数据集
    
    Args:
        dims: 特征维度
        test_datasets: 测试数据集名称列表
        num_hops: 传播跳数
        
    Returns:
        测试数据集列表
    """
    print(f'加载 {len(test_datasets)} 个测试数据集: {test_datasets}')
    
    # 加载数据集
    data_test = [Dataset(dims, name) for name in test_datasets]
    
    # 特征传播
    print(f'进行 {num_hops} 跳特征传播...')
    for te_data in data_test:
        te_data.propagated(num_hops)
    
    return data_test


def inference_on_dataset(model,
                        test_data,
                        dataset_name: str,
                        device: str = 'cuda:0') -> dict:
    """
    在单个数据集上进行推理
    
    Args:
        model: 训练好的模型
        test_data: 测试数据
        dataset_name: 数据集名称
        device: 设备
        
    Returns:
        评估结果字典
    """
    import torch
    
    model.eval()
    
    with torch.no_grad():
        test_graph = test_data.graph.to(device)
        labels = test_graph.ano_labels
        
        query_labels = labels.to(device)
        
        model(test_graph)
        score_components = model.get_ego_score_components()
        query_scores = score_components['total']
        evaluation_mask = getattr(test_graph, 'evaluation_mask', None)
        if evaluation_mask is not None:
            query_labels = query_labels[evaluation_mask]
            query_scores = query_scores[evaluation_mask]
        
        test_score = test_eval(query_labels, query_scores)
    
    result = {
        'dataset': dataset_name,
        'AUROC': test_score['AUROC'],
        'AUPRC': test_score['AUPRC'],
        'num_samples': len(query_labels),
        'num_anomalies': int(query_labels.sum().item()),
        'score_stats': {
            'total_mean': float(score_components['total'].mean().item()),
            'adhesion_mean': float(score_components['adhesion'].mean().item()),
            'scale_mean': float(score_components['scale'].mean().item()),
        }
    }
    
    return result


def run_inference(checkpoint_path: str,
                 test_datasets: list = None,
                 device: str = 'cuda:0',
                 output_dir: str = None,
                 seed: int = None) -> list:
    """
    运行推理
    
    Args:
        checkpoint_path: 检查点路径
        test_datasets: 测试数据集名称列表
        device: 设备
        output_dir: 结果保存目录
        seed: 随机种子
        
    Returns:
        推理结果列表
    """
    print(f'\n{"="*70}')
    print('开始推理')
    print(f'{"="*70}')
    print(f'检查点路径: {checkpoint_path}')
    print(f'设备: {device}')
    print(f'{"="*70}\n')
    
    # 加载模型
    print('加载模型...')
    model, train_config, model_config, checkpoint_info = load_model(checkpoint_path, device)
    
    print(f'\n检查点信息:')
    print(f'  训练轮数: {checkpoint_info["epoch"]}')
    print(f'  试验编号: {checkpoint_info["trial"]}')
    print(f'  时间戳: {checkpoint_info["timestamp"]}')
    if checkpoint_info['seed'] is not None:
        print(f'  随机种子: {checkpoint_info["seed"]}')
    print()
    
    # 设置随机种子
    if seed is not None:
        set_seed(seed)
    elif checkpoint_info['seed'] is not None:
        set_seed(checkpoint_info['seed'])
    
    # 使用默认测试数据集
    if test_datasets is None:
        test_datasets = ['cora', 'citeseer', 'ACM', 'BlogCatalog', 'Facebook',
                        'weibo', 'Reddit', 'Amazon', 'tfinance']
    
    # 准备测试数据集
    dims = model_config.dims
    num_hops = model_config.num_hops

    data_test = prepare_test_datasets(
        dims=dims,
        test_datasets=test_datasets,
        num_hops=num_hops
    )
    
    # 推理
    print(f'\n{"="*70}')
    print('推理中...')
    print(f'{"="*70}\n')
    
    results = []
    for idx, (test_data, dataset_name) in enumerate(zip(data_test, test_datasets)):
        print(f'[{idx+1}/{len(test_datasets)}] 推理数据集: {dataset_name}')
        result = inference_on_dataset(model, test_data, dataset_name, device)
        results.append(result)
        print(f'  AUROC: {result["AUROC"]:.4f}, AUPRC: {result["AUPRC"]:.4f}')
        print(f'  样本数: {result["num_samples"]}, 异常数: {result["num_anomalies"]}\n')
        print(f'  Score Means: total={result["score_stats"]["total_mean"]:.6f}, '
              f'adhesion={result["score_stats"]["adhesion_mean"]:.6f}, '
              f'scale={result["score_stats"]["scale_mean"]:.6f}\n')
    
    # 保存结果
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'inference_results.json')
        
        output_data = {
            'checkpoint_path': checkpoint_path,
            'checkpoint_info': checkpoint_info,
            'model_config': model_config,
            'results': results
        }
        
        with open(result_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f'结果已保存到: {result_file}')
    
    # 打印汇总
    print(f'\n{"="*70}')
    print('推理结果汇总')
    print(f'{"="*70}\n')
    
    for result in results:
        print(f'{"-"*25} {result["dataset"]} {"-"*25}')
        print(f'AUROC: {result["AUROC"]:.4f}')
        print(f'AUPRC: {result["AUPRC"]:.4f}')
        print()
    
    # 计算平均值
    avg_auroc = np.mean([r['AUROC'] for r in results])
    avg_auprc = np.mean([r['AUPRC'] for r in results])
    
    print(f'{"-"*25} 平均值 {"-"*25}')
    print(f'平均 AUROC: {avg_auroc:.4f}')
    print(f'平均 AUPRC: {avg_auprc:.4f}')
    print()
    
    return results


def batch_inference(checkpoint_dir: str,
                   test_datasets: list = None,
                   device: str = 'cuda:0',
                   output_dir: str = None):
    """
    批量推理（对一个模型的多个trial进行推理）
    
    Args:
        checkpoint_dir: 检查点目录（包含多个trial）
        test_datasets: 测试数据集名称列表
        device: 设备
        output_dir: 结果保存目录
    """
    from model_checkpoint import ModelCheckpoint
    
    checkpoint_manager = ModelCheckpoint()
    
    # 获取所有检查点
    model_name = os.path.basename(checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints(model_name)
    
    if not checkpoints:
        print(f'在 {checkpoint_dir} 中没有找到检查点')
        return
    
    print(f'找到 {len(checkpoints)} 个检查点')
    
    all_results = []
    for idx, checkpoint_path in enumerate(checkpoints):
        print(f'\n{"#"*70}')
        print(f'处理检查点 {idx+1}/{len(checkpoints)}')
        print(f'{"#"*70}\n')
        
        results = run_inference(
            checkpoint_path=checkpoint_path,
            test_datasets=test_datasets,
            device=device,
            output_dir=None
        )
        all_results.append(results)
    
    # 聚合结果
    print(f'\n{"="*70}')
    print(f'聚合 {len(all_results)} 个trial的结果')
    print(f'{"="*70}\n')
    
    if test_datasets is None:
        test_datasets = ['cora', 'citeseer', 'ACM', 'BlogCatalog', 'Facebook',
                        'weibo', 'Reddit', 'Amazon', 'tfinance']
    
    for dataset_idx, dataset_name in enumerate(test_datasets):
        aurocs = [trial_results[dataset_idx]['AUROC'] for trial_results in all_results]
        auprcs = [trial_results[dataset_idx]['AUPRC'] for trial_results in all_results]
        
        print(f'{"-"*25} {dataset_name} {"-"*25}')
        print(f'AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}')
        print(f'AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}')
        print()
    
    # 保存聚合结果
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'batch_inference_results.json')
        
        aggregated_results = []
        for dataset_idx, dataset_name in enumerate(test_datasets):
            aurocs = [trial_results[dataset_idx]['AUROC'] for trial_results in all_results]
            auprcs = [trial_results[dataset_idx]['AUPRC'] for trial_results in all_results]
            
            aggregated_results.append({
                'dataset': dataset_name,
                'AUROC_mean': float(np.mean(aurocs)),
                'AUROC_std': float(np.std(aurocs)),
                'AUPRC_mean': float(np.mean(auprcs)),
                'AUPRC_std': float(np.std(auprcs)),
                'all_AUROC': [float(x) for x in aurocs],
                'all_AUPRC': [float(x) for x in auprcs],
            })
        
        with open(result_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        
        print(f'聚合结果已保存到: {result_file}')


def main():
    """主函数"""
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description='recap模型推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点路径或目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='推理设备')
    parser.add_argument('--output-dir', type=str, default='./inference_results', help='结果保存目录')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（可选）')
    parser.add_argument('--batch', action='store_true', help='批量推理模式（对所有trial）')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='测试数据集名称列表（可选）')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量推理模式
        batch_inference(
            checkpoint_dir=args.checkpoint,
            test_datasets=args.datasets,
            device=args.device,
            output_dir=args.output_dir
        )
    else:
        # 单个检查点推理
        run_inference(
            checkpoint_path=args.checkpoint,
            test_datasets=args.datasets,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed
        )
    
    print(f'\n{"="*70}')
    print('推理完成！')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
