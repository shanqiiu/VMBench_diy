# -*- coding: utf-8 -*-
"""
��Ƶģ�������ӻ�����
�ṩģ��������Ŀ��ӻ�չʾ
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from pathlib import Path
import argparse


class BlurVisualization:
    """ģ����������ӻ�����"""
    
    def __init__(self, output_dir: str = "./blur_visualization_results"):
        """
        ��ʼ�����ӻ�����
        
        Args:
            output_dir: ���Ŀ¼
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # ����matplotlib��������
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # ����seaborn��ʽ
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def visualize_quality_scores(self, video_path: str, quality_scores: List[float], 
                                blur_frames: List[int], threshold: float, 
                                save_path: str = None) -> str:
        """
        ���ӻ����������仯
        
        Args:
            video_path: ��Ƶ·��
            quality_scores: ���������б�
            blur_frames: ģ��֡����
            threshold: �����ֵ
            save_path: ����·��
            
        Returns:
            ������ļ�·��
        """
        if save_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(self.output_dir, f"{video_name}_quality_scores.png")
        
        # ����ͼ��
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # ����������������
        frames = list(range(len(quality_scores)))
        ax1.plot(frames, quality_scores, 'b-', linewidth=2, label='��������')
        ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'�����ֵ ({threshold:.3f})')
        
        # ���ģ��֡
        if blur_frames:
            blur_scores = [quality_scores[i] for i in blur_frames]
            ax1.scatter(blur_frames, blur_scores, color='red', s=50, zorder=5, label=f'ģ��֡ ({len(blur_frames)}��)')
        
        ax1.set_xlabel('֡���')
        ax1.set_ylabel('��������')
        ax1.set_title(f'��Ƶ���������仯 - {os.path.basename(video_path)}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # �������������ֲ�ֱ��ͼ
        ax2.hist(quality_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'�����ֵ ({threshold:.3f})')
        ax2.axvline(x=np.mean(quality_scores), color='g', linestyle='-', linewidth=2, label=f'ƽ������ ({np.mean(quality_scores):.3f})')
        
        ax2.set_xlabel('��������')
        ax2.set_ylabel('Ƶ��')
        ax2.set_title('���������ֲ�')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_blur_frames(self, video_path: str, blur_frames: List[int], 
                             num_samples: int = 6, save_path: str = None) -> str:
        """
        ���ӻ�ģ��֡
        
        Args:
            video_path: ��Ƶ·��
            blur_frames: ģ��֡����
            num_samples: ��ʾ��������
            save_path: ����·��
            
        Returns:
            ������ļ�·��
        """
        if save_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(self.output_dir, f"{video_name}_blur_frames.png")
        
        # ��ȡ��Ƶ
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # ��ȡ����֡
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        
        if not frames:
            print("�޷���ȡ��Ƶ֡")
            return ""
        
        # ѡ��Ҫ��ʾ��ģ��֡
        if len(blur_frames) > num_samples:
            selected_frames = np.linspace(0, len(blur_frames)-1, num_samples, dtype=int)
            selected_blur_frames = [blur_frames[i] for i in selected_frames]
        else:
            selected_blur_frames = blur_frames
        
        # ������ͼ
        cols = 3
        rows = (len(selected_blur_frames) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, frame_idx in enumerate(selected_blur_frames):
            row = i // cols
            col = i % cols
            
            if frame_idx < len(frames):
                axes[row, col].imshow(frames[frame_idx])
                axes[row, col].set_title(f'ģ��֡ {frame_idx}')
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        # ���ض������ͼ
        for i in range(len(selected_blur_frames), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'��⵽��ģ��֡ - {os.path.basename(video_path)}', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_batch_results(self, results: List[Dict], save_path: str = None) -> str:
        """
        ���ӻ����������
        
        Args:
            results: ������б�
            save_path: ����·��
            
        Returns:
            ������ļ�·��
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "batch_results_visualization.png")
        
        # ��ȡ����
        video_names = [os.path.basename(r.get('video_path', '')) for r in results]
        confidences = [r.get('confidence', 0.0) for r in results]
        blur_detected = [r.get('blur_detected', False) for r in results]
        blur_ratios = [r.get('blur_ratio', 0.0) for r in results]
        
        # ����ͼ��
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. ���Ŷȷֲ�
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('���Ŷ�')
        ax1.set_ylabel('Ƶ��')
        ax1.set_title('ģ��������Ŷȷֲ�')
        ax1.grid(True, alpha=0.3)
        
        # 2. ģ�������ֲ�
        ax2.hist(blur_ratios, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('ģ������')
        ax2.set_ylabel('Ƶ��')
        ax2.set_title('ģ�������ֲ�')
        ax2.grid(True, alpha=0.3)
        
        # 3. ���Ŷ� vs ģ������ɢ��ͼ
        colors = ['red' if detected else 'blue' for detected in blur_detected]
        ax3.scatter(confidences, blur_ratios, c=colors, alpha=0.6, s=50)
        ax3.set_xlabel('���Ŷ�')
        ax3.set_ylabel('ģ������')
        ax3.set_title('���Ŷ� vs ģ������')
        ax3.grid(True, alpha=0.3)
        
        # ���ͼ��
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='��⵽ģ��'),
                          Patch(facecolor='blue', label='δ��⵽ģ��')]
        ax3.legend(handles=legend_elements)
        
        # 4. �����ͳ�Ʊ�ͼ
        blur_count = sum(blur_detected)
        no_blur_count = len(blur_detected) - blur_count
        
        labels = ['��⵽ģ��', 'δ��⵽ģ��']
        sizes = [blur_count, no_blur_count]
        colors = ['lightcoral', 'lightblue']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('ģ�������ͳ��')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_detection_report(self, result: Dict, save_path: str = None) -> str:
        """
        ������ⱨ��
        
        Args:
            result: ������Ƶ�ļ����
            save_path: ����·��
            
        Returns:
            ������ļ�·��
        """
        if save_path is None:
            video_name = os.path.splitext(os.path.basename(result.get('video_path', '')))[0]
            save_path = os.path.join(self.output_dir, f"{video_name}_detection_report.png")
        
        # ����ͼ��
        fig = plt.figure(figsize=(16, 12))
        
        # ������
        fig.suptitle(f'��Ƶģ����ⱨ�� - {os.path.basename(result.get("video_path", ""))}', 
                    fontsize=20, fontweight='bold')
        
        # �������񲼾�
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. ���������
        ax1 = fig.add_subplot(gs[0, 0])
        blur_detected = result.get('blur_detected', False)
        confidence = result.get('confidence', 0.0)
        
        # �������ָʾ��
        color = 'red' if blur_detected else 'green'
        ax1.text(0.5, 0.5, f'�����: {"��⵽ģ��" if blur_detected else "δ��⵽ģ��"}', 
                ha='center', va='center', fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # 2. ���Ŷ���ʾ
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(['���Ŷ�'], [confidence], color='skyblue', alpha=0.7)
        ax2.set_ylabel('���Ŷ�')
        ax2.set_ylim(0, 1)
        ax2.set_title('������Ŷ�')
        
        # 3. ģ�����س̶�
        ax3 = fig.add_subplot(gs[0, 2])
        severity = result.get('blur_severity', 'δ֪')
        severity_colors = {'��ģ��': 'green', '��΢ģ��': 'yellow', '�е�ģ��': 'orange', '����ģ��': 'red'}
        color = severity_colors.get(severity, 'gray')
        
        ax3.text(0.5, 0.5, f'���س̶�:\n{severity}', ha='center', va='center', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # 4. �ؼ�ָ��
        ax4 = fig.add_subplot(gs[1, :])
        metrics = {
            'ģ������': result.get('blur_ratio', 0.0),
            'ģ��֡��': result.get('blur_frame_count', 0),
            '��֡��': result.get('total_frames', 0),
            'ƽ������': result.get('avg_quality', 0.0),
            '��������½�': result.get('max_quality_drop', 0.0)
        }
        
        bars = ax4.bar(metrics.keys(), metrics.values(), color='lightblue', alpha=0.7)
        ax4.set_ylabel('��ֵ')
        ax4.set_title('�ؼ����ָ��')
        ax4.tick_params(axis='x', rotation=45)
        
        # �����ֵ��ǩ
        for bar, value in zip(bars, metrics.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 5. ����
        ax5 = fig.add_subplot(gs[2, :])
        recommendations = result.get('recommendations', [])
        if recommendations:
            rec_text = '\n'.join([f'? {rec}' for rec in recommendations])
        else:
            rec_text = '�����⽨��'
        
        ax5.text(0.05, 0.95, f'�Ľ�����:\n{rec_text}', transform=ax5.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='lightyellow', alpha=0.8))
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


def main():
    """������"""
    parser = argparse.ArgumentParser(description="��Ƶģ�������ӻ�����")
    parser.add_argument("--results_file", type=str, required=True, help="�����JSON�ļ�·��")
    parser.add_argument("--output_dir", type=str, default="./blur_visualization_results", help="���Ŀ¼")
    parser.add_argument("--video_path", type=str, help="������Ƶ·����������ϸ���ӻ���")
    
    args = parser.parse_args()
    
    # ��ʼ�����ӻ�����
    visualizer = BlurVisualization(args.output_dir)
    
    # ��ȡ�����
    with open(args.results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if isinstance(results, list):
        # ����������ӻ�
        print("����������������ӻ�...")
        batch_viz_path = visualizer.visualize_batch_results(results)
        print(f"����������ӻ��ѱ��浽: {batch_viz_path}")
        
        # Ϊÿ����Ƶ������ϸ����
        for result in results:
            if result.get('video_path'):
                print(f"���� {os.path.basename(result['video_path'])} ����ϸ����...")
                report_path = visualizer.create_detection_report(result)
                print(f"��ϸ�����ѱ��浽: {report_path}")
    
    elif isinstance(results, dict):
        # ����������ӻ�
        print("���ɵ�����������ӻ�...")
        report_path = visualizer.create_detection_report(results)
        print(f"��ⱨ���ѱ��浽: {report_path}")
    
    print(f"���п��ӻ�����ѱ��浽: {args.output_dir}")


if __name__ == "__main__":
    main()
