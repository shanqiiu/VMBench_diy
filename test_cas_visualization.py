# -*- coding: utf-8 -*-
"""
CAS注意力可视化测试脚本
演示如何使用增强版CAS评分系统
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.getcwd(), "VideoMAEv2"))

from enhanced_commonsense_adherence_score import EnhancedCASScorer, get_args
import torch


def test_single_video_visualization():
    """测试单视频注意力可视化"""
    
    print("=== 测试单视频注意力可视化 ===")
    
    # 模拟参数
    class Args:
        def __init__(self):
            self.model = 'vit_base_patch16_224'
            self.input_size = 224
            self.num_frames = 16
            self.tubelet_size = 2
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.enable_visualization = True
            self.visualization_threshold = 0.5
            self.output_dir = './test_visualization_results'
    
    args = Args()
    
    # 创建模拟模型（实际使用时需要加载真实模型）
    print("初始化模型...")
    # model = create_model(args.model, ...)  # 实际使用时取消注释
    model = None  # 模拟模型
    
    if model is None:
        print("警告: 使用模拟模型进行演示")
        print("实际使用时请加载真实的VideoMAEv2模型")
        return
    
    # 创建CAS评分器
    cas_scorer = EnhancedCASScorer(args, model, args.device)
    
    # 测试视频路径（请替换为实际视频路径）
    test_video = "path/to/your/test_video.mp4"
    
    if not os.path.exists(test_video):
        print(f"测试视频不存在: {test_video}")
        print("请将测试视频路径替换为实际存在的视频文件")
        return
    
    print(f"评估视频: {test_video}")
    
    try:
        # 执行评估
        result = cas_scorer.evaluate_with_visualization(test_video)
        
        print(f"CAS评分: {result['cas_score']:.3f}")
        
        if result['visualization']:
            print("? 已生成注意力可视化")
            print(f"  输出目录: {result['visualization']['output_dir']}")
        else:
            print("? CAS评分正常，无需可视化")
            
    except Exception as e:
        print(f"评估过程中出错: {e}")


def test_batch_visualization():
    """测试批量视频可视化"""
    
    print("\n=== 测试批量视频可视化 ===")
    
    # 模拟参数
    class Args:
        def __init__(self):
            self.model = 'vit_base_patch16_224'
            self.input_size = 224
            self.num_frames = 16
            self.tubelet_size = 2
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.enable_visualization = True
            self.visualization_threshold = 0.5
            self.output_dir = './test_batch_visualization_results'
    
    args = Args()
    
    # 创建模拟模型
    model = None  # 模拟模型
    
    if model is None:
        print("警告: 使用模拟模型进行演示")
        return
    
    # 创建CAS评分器
    cas_scorer = EnhancedCASScorer(args, model, args.device)
    
    # 测试视频目录（请替换为实际目录）
    test_video_dir = "path/to/your/video/directory"
    
    if not os.path.exists(test_video_dir):
        print(f"测试视频目录不存在: {test_video_dir}")
        print("请将测试目录路径替换为实际存在的目录")
        return
    
    print(f"批量评估视频目录: {test_video_dir}")
    
    try:
        # 执行批量评估
        results = cas_scorer.batch_evaluate_with_visualization(
            test_video_dir, 
            output_dir=args.output_dir
        )
        
        print(f"? 批量评估完成")
        print(f"  处理视频数量: {len(results)}")
        print(f"  结果保存目录: {args.output_dir}")
        
        # 统计信息
        scores = [r.get('cas_score', 0.0) for r in results if 'error' not in r]
        low_scores = [s for s in scores if s < 0.5]
        visualizations = [r for r in results if r.get('visualization') is not None]
        
        print(f"  平均CAS评分: {sum(scores)/len(scores):.3f}")
        print(f"  低分视频数量: {len(low_scores)}")
        print(f"  生成可视化数量: {len(visualizations)}")
        
    except Exception as e:
        print(f"批量评估过程中出错: {e}")


def demonstrate_attention_analysis():
    """演示注意力分析功能"""
    
    print("\n=== 注意力分析功能演示 ===")
    
    # 模拟注意力权重数据
    import numpy as np
    
    # 创建模拟的注意力热力图
    attention_heatmap = np.random.rand(224, 224)
    
    # 模拟高注意力区域（异常区域）
    attention_heatmap[100:150, 100:150] += 0.5  # 增加某个区域的注意力
    
    print("注意力分析结果:")
    print(f"  注意力图尺寸: {attention_heatmap.shape}")
    print(f"  平均注意力: {np.mean(attention_heatmap):.3f}")
    print(f"  最大注意力: {np.max(attention_heatmap):.3f}")
    print(f"  注意力标准差: {np.std(attention_heatmap):.3f}")
    
    # 找出高注意力区域
    threshold = np.percentile(attention_heatmap, 90)
    high_attention_regions = np.where(attention_heatmap > threshold)
    
    print(f"  高注意力区域数量: {len(high_attention_regions[0])}")
    print(f"  高注意力阈值: {threshold:.3f}")
    
    # 模拟CAS评分
    cas_score = 0.3  # 低分
    
    print(f"\nCAS评分分析:")
    print(f"  CAS评分: {cas_score:.3f}")
    
    if cas_score < 0.5:
        print("  ? 检测到低CAS评分，可能存在常识违反")
        print("  ? 建议检查高注意力区域以了解异常原因")
    else:
        print("  ? CAS评分正常，未检测到明显异常")


def create_usage_example():
    """创建使用示例"""
    
    example_code = '''
# CAS注意力可视化使用示例

# 1. 单视频评估
from enhanced_commonsense_adherence_score import EnhancedCASScorer
import torch

# 初始化参数
args = {
    'model': 'vit_base_patch16_224',
    'input_size': 224,
    'num_frames': 16,
    'tubelet_size': 2,
    'device': 'cuda',
    'enable_visualization': True,
    'visualization_threshold': 0.5
}

# 加载模型
model = create_model(args['model'], ...)
model.to(args['device'])

# 创建CAS评分器
cas_scorer = EnhancedCASScorer(args, model, args['device'])

# 评估单个视频
result = cas_scorer.evaluate_with_visualization('path/to/video.mp4')
print(f"CAS评分: {result['cas_score']:.3f}")

# 2. 批量评估
results = cas_scorer.batch_evaluate_with_visualization(
    'path/to/video/directory',
    output_dir='./visualization_results'
)

# 3. 命令行使用
# python enhanced_commonsense_adherence_score.py \\
#     --video_path path/to/video.mp4 \\
#     --enable_visualization \\
#     --output_dir ./results

# python enhanced_commonsense_adherence_score.py \\
#     --video_dir path/to/videos \\
#     --enable_visualization \\
#     --output_dir ./batch_results
'''
    
    print("\n=== 使用示例 ===")
    print(example_code)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CAS注意力可视化测试')
    parser.add_argument('--test_mode', choices=['single', 'batch', 'demo', 'example'], 
                       default='demo', help='测试模式')
    
    args = parser.parse_args()
    
    print("CAS注意力权重可视化测试")
    print("=" * 50)
    
    if args.test_mode == 'single':
        test_single_video_visualization()
    elif args.test_mode == 'batch':
        test_batch_visualization()
    elif args.test_mode == 'demo':
        demonstrate_attention_analysis()
    elif args.test_mode == 'example':
        create_usage_example()
    
    print("\n测试完成！")
    print("\n注意事项:")
    print("1. 实际使用时需要加载真实的VideoMAEv2模型")
    print("2. 请确保视频文件路径正确")
    print("3. 可视化结果将保存到指定输出目录")
    print("4. 低CAS评分的视频会自动生成注意力可视化")


if __name__ == '__main__':
    main()
