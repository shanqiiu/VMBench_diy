# -*- coding: utf-8 -*-
"""
CASע����Ȩ�ؿ��ӻ�ģ��
����VideoMAEv2��ע��������ʵ�ֿ��ӻ�����
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
    """ע����Ȩ����ȡ����"""
    
    def __init__(self):
        self.attention_weights = []
        self.layer_names = []
    
    def __call__(self, module, input, output):
        """���Ӻ�������ȡע����Ȩ��"""
        if hasattr(module, 'attn') and hasattr(module.attn, 'attention_weights'):
            # ��ȡע����Ȩ��
            attn_weights = module.attn.attention_weights
            self.attention_weights.append(attn_weights.detach().cpu())
            self.layer_names.append(module.__class__.__name__)


class CASAttentionVisualizer:
    """CASע����Ȩ�ؿ��ӻ���"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.attention_hook = AttentionHook()
        self.patch_size = 16  # VideoMAEv2Ĭ��patch��С
        self.num_frames = 16  # Ĭ��֡��
        
    def register_attention_hooks(self):
        """ע��ע�������ӵ�ģ��"""
        hooks = []
        
        # Ϊÿ��Transformer Blockע�ṳ��
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention) or 'attn' in name.lower():
                hook = module.register_forward_hook(self.attention_hook)
                hooks.append(hook)
        
        return hooks
    
    def extract_attention_weights(self, video_tensor):
        """��ȡע����Ȩ��"""
        self.model.eval()
        self.attention_hook.attention_weights = []
        self.attention_hook.layer_names = []
        
        # ע�ṳ��
        hooks = self.register_attention_hooks()
        
        try:
            with torch.no_grad():
                # ǰ�򴫲�
                outputs = self.model(video_tensor)
                
                # ��ȡע����Ȩ��
                attention_weights = self.attention_hook.attention_weights
                layer_names = self.attention_hook.layer_names
                
                return attention_weights, layer_names, outputs
                
        finally:
            # �Ƴ�����
            for hook in hooks:
                hook.remove()
    
    def compute_attention_rollout(self, attention_weights, layer_names):
        """����ע����rollout"""
        # ѡ����󼸲��ע����Ȩ��
        last_layers = attention_weights[-4:]  # ȡ���4��
        
        # ƽ������ע����ͷ
        averaged_attention = []
        for layer_attn in last_layers:
            # layer_attn shape: [batch, num_heads, num_patches, num_patches]
            avg_attn = layer_attn.mean(dim=1)  # ƽ������ͷ
            averaged_attention.append(avg_attn)
        
        # ����rollout
        rollout = torch.eye(averaged_attention[0].shape[-1])
        for attn in averaged_attention:
            rollout = torch.matmul(attn, rollout)
        
        return rollout
    
    def create_attention_heatmap(self, attention_weights, video_frames, 
                                method='rollout', layer_idx=-1):
        """����ע��������ͼ"""
        
        if method == 'rollout':
            # ʹ��rollout����
            rollout = self.compute_attention_rollout(attention_weights, [])
            attention_map = rollout[0, 1:]  # �ų�CLS token
        else:
            # ʹ�õ���ע����
            layer_attn = attention_weights[layer_idx]
            attention_map = layer_attn[0, :, 0, 1:]  # [num_heads, num_patches]
            attention_map = attention_map.mean(dim=0)  # ƽ������ͷ
        
        # ����Ϊ�ռ�ά��
        num_patches = attention_map.shape[0]
        num_spatial_patches = int(np.sqrt(num_patches))
        
        if num_spatial_patches * num_spatial_patches != num_patches:
            # �����������patch�����
            h_patches = int(np.sqrt(num_patches))
            w_patches = num_patches // h_patches
        else:
            h_patches = w_patches = num_spatial_patches
        
        # ����ע����ͼ
        attention_map = attention_map.reshape(h_patches, w_patches)
        
        # �ϲ�����ԭͼ�ߴ�
        target_size = (224, 224)  # VideoMAEv2Ĭ������ߴ�
        attention_heatmap = cv2.resize(
            attention_map.numpy(), 
            target_size, 
            interpolation=cv2.INTER_CUBIC
        )
        
        return attention_heatmap
    
    def visualize_attention_on_frames(self, video_frames, attention_heatmap, 
                                    alpha=0.6, colormap='jet'):
        """����Ƶ֡�Ͽ��ӻ�ע����"""
        visualized_frames = []
        
        for frame in video_frames:
            # ת��֡��ʽ
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
            
            # ��һ��ע����ͼ
            attention_norm = (attention_heatmap - attention_heatmap.min()) / \
                           (attention_heatmap.max() - attention_heatmap.min())
            
            # Ӧ����ɫӳ��
            if colormap == 'jet':
                attention_colored = plt.cm.jet(attention_norm)[:, :, :3]
                attention_colored = (attention_colored * 255).astype(np.uint8)
            else:
                attention_colored = cv2.applyColorMap(
                    (attention_norm * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
            
            # ����ע����ͼ��ԭ֡
            blended = cv2.addWeighted(frame, 1-alpha, attention_colored, alpha, 0)
            visualized_frames.append(blended)
        
        return visualized_frames
    
    def generate_attention_report(self, video_path, cas_score, attention_weights, 
                                output_dir='./attention_visualization'):
        """����ע�������ӻ�����"""
        
        # �������Ŀ¼
        os.makedirs(output_dir, exist_ok=True)
        
        # ������Ƶ֡
        video_frames = self.load_video_frames(video_path)
        
        # ����ע��������ͼ
        attention_heatmap = self.create_attention_heatmap(
            attention_weights, video_frames, method='rollout'
        )
        
        # ���ӻ�ע����
        visualized_frames = self.visualize_attention_on_frames(
            video_frames, attention_heatmap
        )
        
        # ������ӻ����
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
        """����ע�������ӻ����"""
        
        # 1. ����ע��������ͼ
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_heatmap, cmap='jet')
        plt.colorbar()
        plt.title(f'CAS Attention Heatmap (Score: {cas_score:.3f})')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'attention_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ������ӻ���Ƶ֡
        for i, frame in enumerate(visualized_frames):
            cv2.imwrite(
                os.path.join(output_dir, f'frame_{i:03d}_attention.jpg'), 
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
        
        # 3. ����ע����ͳ�Ʊ���
        self.create_attention_report(attention_heatmap, cas_score, output_dir)
    
    def create_attention_report(self, attention_heatmap, cas_score, output_dir):
        """����ע������������"""
        
        # ����ע����ͳ��
        attention_stats = {
            'mean_attention': float(np.mean(attention_heatmap)),
            'max_attention': float(np.max(attention_heatmap)),
            'min_attention': float(np.min(attention_heatmap)),
            'std_attention': float(np.std(attention_heatmap)),
            'cas_score': float(cas_score)
        }
        
        # �ҳ���ע��������
        threshold = np.percentile(attention_heatmap, 90)
        high_attention_regions = np.where(attention_heatmap > threshold)
        
        # ���ɱ���
        report = f"""
# CASע������������

## ������Ϣ
- CAS����: {cas_score:.3f}
- ע����ͼ�ߴ�: {attention_heatmap.shape}

## ע����ͳ��
- ƽ��ע����: {attention_stats['mean_attention']:.3f}
- ���ע����: {attention_stats['max_attention']:.3f}
- ��Сע����: {attention_stats['min_attention']:.3f}
- ע������׼��: {attention_stats['std_attention']:.3f}

## ��ע��������
- ��ע������������: {len(high_attention_regions[0])}
- ��ע������ֵ: {threshold:.3f}

## �������
"""
        
        if cas_score < 0.5:
            report += "- ��⵽��CAS���֣����ܴ��ڳ�ʶΥ��\n"
            report += "- ��ע����������ܰ����쳣����\n"
        else:
            report += "- CAS����������δ��⵽�����쳣\n"
        
        # ���汨��
        with open(os.path.join(output_dir, 'attention_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
    
    def load_video_frames(self, video_path):
        """������Ƶ֡"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ת����ɫ�ռ�
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames


def enhance_cas_with_attention_visualization(video_path, model, device='cuda'):
    """��ǿ��CAS������֧��ע�������ӻ�"""
    
    # ��ʼ�����ӻ���
    visualizer = CASAttentionVisualizer(model, device)
    
    # ������Ƶ
    video_tensor = load_video_tensor(video_path)  # ��Ҫʵ����Ƶ���غ���
    
    # ��ȡע����Ȩ��
    attention_weights, layer_names, cas_outputs = visualizer.extract_attention_weights(video_tensor)
    
    # ����CAS����
    cas_score = compute_cas_score(cas_outputs)  # ��Ҫʵ��CAS���ּ���
    
    # ���ɿ��ӻ�����
    if cas_score < 0.5:  # ֻ�Եͷֽ��п��ӻ�
        visualization_results = visualizer.generate_attention_report(
            video_path, cas_score, attention_weights
        )
        return cas_score, visualization_results
    else:
        return cas_score, None


def load_video_tensor(video_path, num_frames=16, img_size=224):
    """������ƵΪtensor��ʽ"""
    # ������Ҫʵ����Ƶ���غ�Ԥ����
    # ������״Ϊ [1, 3, num_frames, img_size, img_size] ��tensor
    pass


def compute_cas_score(model_outputs):
    """����CAS����"""
    # ������Ҫʵ��CAS���ּ����߼�
    # ����0-1֮�������
    pass


if __name__ == "__main__":
    # ʹ��ʾ��
    print("CASע����Ȩ�ؿ��ӻ�ģ���Ѽ���")
    print("ʹ�÷�����")
    print("1. ��ʼ��CASAttentionVisualizer")
    print("2. ����extract_attention_weights��ȡע����Ȩ��")
    print("3. ����generate_attention_report���ɿ��ӻ�����")
