# -*- coding: utf-8 -*-
"""
增强版常识遵循评分系统
集成注意力权重可视化功能
"""

import sys
import argparse
import datetime
import json
import csv
import os
import random
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma

sys.path.insert(0, os.path.join(os.getcwd(), "VideoMAEv2"))

# 导入原有模块
import models  # noqa: F401
import utils
from dataset import build_dataset
from engine_for_finetuning import (
    final_test,
    merge,
    train_one_epoch,
    validation_one_epoch
)
from optim_factory import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate

from bench_utils.cas_utils import final_merge, final_test
from cas_attention_visualizer import CASAttentionVisualizer


class EnhancedCASScorer:
    """增强版CAS评分器，支持注意力可视化"""
    
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        self.visualizer = CASAttentionVisualizer(model, device)
        
    def evaluate_with_visualization(self, video_path, save_visualization=True):
        """带可视化的CAS评估"""
        
        # 1. 执行原有CAS评估
        cas_score = self.original_cas_evaluation(video_path)
        
        # 2. 如果分数较低，进行可视化分析
        if cas_score < 0.5 and save_visualization:
            print(f"检测到低CAS评分 ({cas_score:.3f})，开始生成注意力可视化...")
            
            # 加载视频
            video_tensor = self.load_video_tensor(video_path)
            
            # 提取注意力权重
            attention_weights, layer_names, outputs = self.visualizer.extract_attention_weights(video_tensor)
            
            # 生成可视化报告
            visualization_results = self.visualizer.generate_attention_report(
                video_path, cas_score, attention_weights
            )
            
            return {
                'cas_score': cas_score,
                'visualization': visualization_results,
                'attention_weights': attention_weights,
                'layer_names': layer_names
            }
        else:
            return {
                'cas_score': cas_score,
                'visualization': None
            }
    
    def original_cas_evaluation(self, video_path):
        """原有CAS评估逻辑"""
        # 这里实现原有的CAS评分逻辑
        # 返回0-1之间的评分
        pass
    
    def load_video_tensor(self, video_path, num_frames=16, img_size=224):
        """加载视频为tensor格式"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # 读取视频帧
        frame_count = 0
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 预处理帧
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            
            # 标准化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = (frame - mean) / std
            
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        # 转换为tensor
        video_tensor = torch.from_numpy(np.array(frames)).permute(3, 0, 1, 2)  # [C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]
        
        return video_tensor.to(self.device)
    
    def batch_evaluate_with_visualization(self, video_dir, output_dir='./cas_visualization_results'):
        """批量评估带可视化"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有视频文件
        video_files = []
        for ext in ['.mp4', '.avi', '.mov']:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        results = []
        
        for video_file in video_files:
            print(f"处理视频: {video_file.name}")
            
            try:
                # 执行评估
                result = self.evaluate_with_visualization(str(video_file))
                result['video_path'] = str(video_file)
                results.append(result)
                
                # 保存单个视频结果
                video_output_dir = os.path.join(output_dir, video_file.stem)
                if result['visualization']:
                    print(f"  可视化结果保存到: {video_output_dir}")
                
            except Exception as e:
                print(f"处理视频 {video_file.name} 时出错: {e}")
                results.append({
                    'video_path': str(video_file),
                    'cas_score': 0.0,
                    'error': str(e)
                })
        
        # 保存批量结果
        self.save_batch_results(results, output_dir)
        
        return results
    
    def save_batch_results(self, results, output_dir):
        """保存批量评估结果"""
        
        # 1. 保存JSON结果
        json_path = os.path.join(output_dir, 'cas_visualization_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 2. 保存CSV摘要
        csv_path = os.path.join(output_dir, 'cas_summary.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Video', 'CAS_Score', 'Has_Visualization', 'Error'])
            
            for result in results:
                writer.writerow([
                    Path(result['video_path']).name,
                    result.get('cas_score', 0.0),
                    result.get('visualization') is not None,
                    result.get('error', '')
                ])
        
        # 3. 生成统计报告
        self.generate_statistics_report(results, output_dir)
        
        print(f"批量评估结果已保存到: {output_dir}")
    
    def generate_statistics_report(self, results, output_dir):
        """生成统计报告"""
        
        # 计算统计信息
        scores = [r.get('cas_score', 0.0) for r in results if 'error' not in r]
        low_score_count = sum(1 for s in scores if s < 0.5)
        visualization_count = sum(1 for r in results if r.get('visualization') is not None)
        
        report = f"""
# CAS注意力可视化统计报告

## 基本统计
- 总视频数量: {len(results)}
- 成功处理: {len(scores)}
- 处理失败: {len(results) - len(scores)}

## CAS评分统计
- 平均评分: {np.mean(scores):.3f}
- 最低评分: {np.min(scores):.3f}
- 最高评分: {np.max(scores):.3f}
- 评分标准差: {np.std(scores):.3f}

## 异常检测
- 低分视频数量 (< 0.5): {low_score_count}
- 生成可视化数量: {visualization_count}
- 异常检测率: {low_score_count/len(scores)*100:.1f}%

## 分析结果
"""
        
        if low_score_count > 0:
            report += f"- 检测到 {low_score_count} 个视频存在常识违反\n"
            report += f"- 已为 {visualization_count} 个视频生成注意力可视化\n"
            report += "- 建议检查低分视频的注意力热力图以了解异常原因\n"
        else:
            report += "- 所有视频的CAS评分正常，未检测到明显异常\n"
        
        # 保存报告
        report_path = os.path.join(output_dir, 'statistics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(
        'Enhanced VideoMAE CAS evaluation with attention visualization',
        add_help=False)
    
    # 原有参数
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--device', default='cuda', help='device to use')
    
    # 可视化相关参数
    parser.add_argument('--enable_visualization', action='store_true', 
                       help='Enable attention visualization')
    parser.add_argument('--visualization_threshold', type=float, default=0.5,
                       help='Threshold for generating visualizations')
    parser.add_argument('--output_dir', default='./cas_visualization_results',
                       help='Output directory for visualizations')
    parser.add_argument('--video_path', type=str, help='Single video path for evaluation')
    parser.add_argument('--video_dir', type=str, help='Directory containing videos for batch evaluation')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = get_args()
    
    # 初始化设备
    device = torch.device(args.device)
    
    # 创建模型
    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        num_classes=400,  # 根据实际需求调整
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
    )
    
    model.to(device)
    
    # 创建增强版CAS评分器
    cas_scorer = EnhancedCASScorer(args, model, device)
    
    if args.video_path:
        # 单视频评估
        print(f"评估单个视频: {args.video_path}")
        result = cas_scorer.evaluate_with_visualization(
            args.video_path, 
            save_visualization=args.enable_visualization
        )
        
        print(f"CAS评分: {result['cas_score']:.3f}")
        if result['visualization']:
            print(f"可视化结果已保存")
            
    elif args.video_dir:
        # 批量评估
        print(f"批量评估视频目录: {args.video_dir}")
        results = cas_scorer.batch_evaluate_with_visualization(
            args.video_dir, 
            output_dir=args.output_dir
        )
        
        print(f"批量评估完成，共处理 {len(results)} 个视频")
        
    else:
        print("请指定 --video_path 或 --video_dir 参数")


if __name__ == '__main__':
    main()
