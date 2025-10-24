# -*- coding: utf-8 -*-
"""
CAS注意力权重可视化模块
基于VideoMAEv2的注意力机制实现可视化功能
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path


class AttentionHook:
    """注意力权重提取钩子"""
    
    def __init__(self):
        self.attention_weights = []
        self.layer_names = []
    
    def __call__(self, module, input, output):
        """钩子函数，提取注意力权重"""
        if hasattr(module, 'attn') and hasattr(module.attn, 'attention_weights'):
            # 提取注意力权重
            attn_weights = module.attn.attention_weights
            self.attention_weights.append(attn_weights.detach().cpu())
            self.layer_names.append(module.__class__.__name__)


class CASAttentionVisualizer:
    """CAS注意力权重可视化器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.attention_hook = AttentionHook()
        self.patch_size = 16  # VideoMAEv2默认patch大小
        self.num_frames = 16  # 默认帧数
        
    def register_attention_hooks(self):
        """注册注意力钩子到模型"""
        hooks = []
        
        # 为每个Transformer Block注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention) or 'attn' in name.lower():
                hook = module.register_forward_hook(self.attention_hook)
                hooks.append(hook)
        
        return hooks
    
    def extract_attention_weights(self, video_tensor):
        """提取注意力权重"""
        self.model.eval()
        self.attention_hook.attention_weights = []
        self.attention_hook.layer_names = []
        
        # 注册钩子
        hooks = self.register_attention_hooks()
        
        try:
            with torch.no_grad():
                # 前向传播
                outputs = self.model(video_tensor)
                
                # 提取注意力权重
                attention_weights = self.attention_hook.attention_weights
                layer_names = self.attention_hook.layer_names
                
                return attention_weights, layer_names, outputs
                
        finally:
            # 移除钩子
            for hook in hooks:
                hook.remove()
    
    def compute_attention_rollout(self, attention_weights, layer_names):
        """计算注意力rollout"""
        # 选择最后几层的注意力权重
        last_layers = attention_weights[-4:]  # 取最后4层
        
        # 平均所有注意力头
        averaged_attention = []
        for layer_attn in last_layers:
            # layer_attn shape: [batch, num_heads, num_patches, num_patches]
            avg_attn = layer_attn.mean(dim=1)  # 平均所有头
            averaged_attention.append(avg_attn)
        
        # 计算rollout
        rollout = torch.eye(averaged_attention[0].shape[-1])
        for attn in averaged_attention:
            rollout = torch.matmul(attn, rollout)
        
        return rollout
    
    def create_attention_heatmap(self, attention_weights, video_frames, 
                                method='rollout', layer_idx=-1):
        """创建注意力热力图"""
        
        if method == 'rollout':
            # 使用rollout方法
            rollout = self.compute_attention_rollout(attention_weights, [])
            attention_map = rollout[0, 1:]  # 排除CLS token
        else:
            # 使用单层注意力
            layer_attn = attention_weights[layer_idx]
            attention_map = layer_attn[0, :, 0, 1:]  # [num_heads, num_patches]
            attention_map = attention_map.mean(dim=0)  # 平均所有头
        
        # 重塑为空间维度
        num_patches = attention_map.shape[0]
        num_spatial_patches = int(np.sqrt(num_patches))
        
        if num_spatial_patches * num_spatial_patches != num_patches:
            # 处理非正方形patch的情况
            h_patches = int(np.sqrt(num_patches))
            w_patches = num_patches // h_patches
        else:
            h_patches = w_patches = num_spatial_patches
        
        # 重塑注意力图
        attention_map = attention_map.reshape(h_patches, w_patches)
        
        # 上采样到原图尺寸
        target_size = (224, 224)  # VideoMAEv2默认输入尺寸
        attention_heatmap = cv2.resize(
            attention_map.numpy(), 
            target_size, 
            interpolation=cv2.INTER_CUBIC
        )
        
        return attention_heatmap
    
    def visualize_attention_on_frames(self, video_frames, attention_heatmap, 
                                    alpha=0.6, colormap='jet'):
        """在视频帧上可视化注意力"""
        visualized_frames = []
        
        for frame in video_frames:
            # 转换帧格式
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
            
            # 归一化注意力图
            attention_norm = (attention_heatmap - attention_heatmap.min()) / \
                           (attention_heatmap.max() - attention_heatmap.min())
            
            # 应用颜色映射
            if colormap == 'jet':
                attention_colored = plt.cm.jet(attention_norm)[:, :, :3]
                attention_colored = (attention_colored * 255).astype(np.uint8)
            else:
                attention_colored = cv2.applyColorMap(
                    (attention_norm * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
            
            # 叠加注意力图到原帧
            blended = cv2.addWeighted(frame, 1-alpha, attention_colored, alpha, 0)
            visualized_frames.append(blended)
        
        return visualized_frames
    
    def generate_attention_report(self, video_path, cas_score, attention_weights, 
                                output_dir='./attention_visualization'):
        """生成注意力可视化报告"""
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载视频帧
        video_frames = self.load_video_frames(video_path)
        
        # 创建注意力热力图
        attention_heatmap = self.create_attention_heatmap(
            attention_weights, video_frames, method='rollout'
        )
        
        # 可视化注意力
        visualized_frames = self.visualize_attention_on_frames(
            video_frames, attention_heatmap
        )
        
        # 保存可视化结果
        self.save_attention_visualization(
            visualized_frames, attention_heatmap, cas_score, output_dir
        )
        
        return {
            'cas_score': cas_score,
            'attention_heatmap': attention_heatmap,
            'visualized_frames': visualized_frames,
            'output_dir': output_dir
        }
    
    def save_attention_visualization(self, visualized_frames, attention_heatmap, 
                                   cas_score, output_dir):
        """保存注意力可视化结果"""
        
        # 1. 保存注意力热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_heatmap, cmap='jet')
        plt.colorbar()
        plt.title(f'CAS Attention Heatmap (Score: {cas_score:.3f})')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'attention_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 保存可视化视频帧
        for i, frame in enumerate(visualized_frames):
            cv2.imwrite(
                os.path.join(output_dir, f'frame_{i:03d}_attention.jpg'), 
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
        
        # 3. 创建注意力统计报告
        self.create_attention_report(attention_heatmap, cas_score, output_dir)
    
    def create_attention_report(self, attention_heatmap, cas_score, output_dir):
        """创建注意力分析报告"""
        
        # 计算注意力统计
        attention_stats = {
            'mean_attention': float(np.mean(attention_heatmap)),
            'max_attention': float(np.max(attention_heatmap)),
            'min_attention': float(np.min(attention_heatmap)),
            'std_attention': float(np.std(attention_heatmap)),
            'cas_score': float(cas_score)
        }
        
        # 找出高注意力区域
        threshold = np.percentile(attention_heatmap, 90)
        high_attention_regions = np.where(attention_heatmap > threshold)
        
        # 生成报告
        report = f"""
# CAS注意力分析报告

## 基本信息
- CAS评分: {cas_score:.3f}
- 注意力图尺寸: {attention_heatmap.shape}

## 注意力统计
- 平均注意力: {attention_stats['mean_attention']:.3f}
- 最大注意力: {attention_stats['max_attention']:.3f}
- 最小注意力: {attention_stats['min_attention']:.3f}
- 注意力标准差: {attention_stats['std_attention']:.3f}

## 高注意力区域
- 高注意力区域数量: {len(high_attention_regions[0])}
- 高注意力阈值: {threshold:.3f}

## 分析结果
"""
        
        if cas_score < 0.5:
            report += "- 检测到低CAS评分，可能存在常识违反\n"
            report += "- 高注意力区域可能包含异常内容\n"
        else:
            report += "- CAS评分正常，未检测到明显异常\n"
        
        # 保存报告
        with open(os.path.join(output_dir, 'attention_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
    
    def load_video_frames(self, video_path):
        """加载视频帧"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换颜色空间
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames


def enhance_cas_with_attention_visualization(video_path, model, device='cuda'):
    """增强版CAS评估，支持注意力可视化"""
    
    # 初始化可视化器
    visualizer = CASAttentionVisualizer(model, device)
    
    # 加载视频
    video_tensor = load_video_tensor(video_path)  # 需要实现视频加载函数
    
    # 提取注意力权重
    attention_weights, layer_names, cas_outputs = visualizer.extract_attention_weights(video_tensor)
    
    # 计算CAS评分
    cas_score = compute_cas_score(cas_outputs)  # 需要实现CAS评分计算
    
    # 生成可视化报告
    if cas_score < 0.5:  # 只对低分进行可视化
        visualization_results = visualizer.generate_attention_report(
            video_path, cas_score, attention_weights
        )
        return cas_score, visualization_results
    else:
        return cas_score, None


def load_video_tensor(video_path, num_frames=16, img_size=224):
    """加载视频为tensor格式"""
    # 这里需要实现视频加载和预处理
    # 返回形状为 [1, 3, num_frames, img_size, img_size] 的tensor
    pass


def compute_cas_score(model_outputs):
    """计算CAS评分"""
    # 这里需要实现CAS评分计算逻辑
    # 返回0-1之间的评分
    pass


if __name__ == "__main__":
    # 使用示例
    print("CAS注意力权重可视化模块已加载")
    print("使用方法：")
    print("1. 初始化CASAttentionVisualizer")
    print("2. 调用extract_attention_weights提取注意力权重")
    print("3. 调用generate_attention_report生成可视化报告")
