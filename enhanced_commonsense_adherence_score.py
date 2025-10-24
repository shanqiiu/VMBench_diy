# -*- coding: utf-8 -*-
"""
��ǿ�泣ʶ��ѭ����ϵͳ
����ע����Ȩ�ؿ��ӻ�����
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

# ����ԭ��ģ��
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
    """��ǿ��CAS��������֧��ע�������ӻ�"""
    
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        self.visualizer = CASAttentionVisualizer(model, device)
        
    def evaluate_with_visualization(self, video_path, save_visualization=True):
        """�����ӻ���CAS����"""
        
        # 1. ִ��ԭ��CAS����
        cas_score = self.original_cas_evaluation(video_path)
        
        # 2. ��������ϵͣ����п��ӻ�����
        if cas_score < 0.5 and save_visualization:
            print(f"��⵽��CAS���� ({cas_score:.3f})����ʼ����ע�������ӻ�...")
            
            # ������Ƶ
            video_tensor = self.load_video_tensor(video_path)
            
            # ��ȡע����Ȩ��
            attention_weights, layer_names, outputs = self.visualizer.extract_attention_weights(video_tensor)
            
            # ���ɿ��ӻ�����
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
        """ԭ��CAS�����߼�"""
        # ����ʵ��ԭ�е�CAS�����߼�
        # ����0-1֮�������
        pass
    
    def load_video_tensor(self, video_path, num_frames=16, img_size=224):
        """������ƵΪtensor��ʽ"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # ��ȡ��Ƶ֡
        frame_count = 0
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ԥ����֡
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            
            # ��׼��
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = (frame - mean) / std
            
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        # ת��Ϊtensor
        video_tensor = torch.from_numpy(np.array(frames)).permute(3, 0, 1, 2)  # [C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]
        
        return video_tensor.to(self.device)
    
    def batch_evaluate_with_visualization(self, video_dir, output_dir='./cas_visualization_results'):
        """�������������ӻ�"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # ��ȡ������Ƶ�ļ�
        video_files = []
        for ext in ['.mp4', '.avi', '.mov']:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        results = []
        
        for video_file in video_files:
            print(f"������Ƶ: {video_file.name}")
            
            try:
                # ִ������
                result = self.evaluate_with_visualization(str(video_file))
                result['video_path'] = str(video_file)
                results.append(result)
                
                # ���浥����Ƶ���
                video_output_dir = os.path.join(output_dir, video_file.stem)
                if result['visualization']:
                    print(f"  ���ӻ�������浽: {video_output_dir}")
                
            except Exception as e:
                print(f"������Ƶ {video_file.name} ʱ����: {e}")
                results.append({
                    'video_path': str(video_file),
                    'cas_score': 0.0,
                    'error': str(e)
                })
        
        # �����������
        self.save_batch_results(results, output_dir)
        
        return results
    
    def save_batch_results(self, results, output_dir):
        """���������������"""
        
        # 1. ����JSON���
        json_path = os.path.join(output_dir, 'cas_visualization_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 2. ����CSVժҪ
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
        
        # 3. ����ͳ�Ʊ���
        self.generate_statistics_report(results, output_dir)
        
        print(f"������������ѱ��浽: {output_dir}")
    
    def generate_statistics_report(self, results, output_dir):
        """����ͳ�Ʊ���"""
        
        # ����ͳ����Ϣ
        scores = [r.get('cas_score', 0.0) for r in results if 'error' not in r]
        low_score_count = sum(1 for s in scores if s < 0.5)
        visualization_count = sum(1 for r in results if r.get('visualization') is not None)
        
        report = f"""
# CASע�������ӻ�ͳ�Ʊ���

## ����ͳ��
- ����Ƶ����: {len(results)}
- �ɹ�����: {len(scores)}
- ����ʧ��: {len(results) - len(scores)}

## CAS����ͳ��
- ƽ������: {np.mean(scores):.3f}
- �������: {np.min(scores):.3f}
- �������: {np.max(scores):.3f}
- ���ֱ�׼��: {np.std(scores):.3f}

## �쳣���
- �ͷ���Ƶ���� (< 0.5): {low_score_count}
- ���ɿ��ӻ�����: {visualization_count}
- �쳣�����: {low_score_count/len(scores)*100:.1f}%

## �������
"""
        
        if low_score_count > 0:
            report += f"- ��⵽ {low_score_count} ����Ƶ���ڳ�ʶΥ��\n"
            report += f"- ��Ϊ {visualization_count} ����Ƶ����ע�������ӻ�\n"
            report += "- ������ͷ���Ƶ��ע��������ͼ���˽��쳣ԭ��\n"
        else:
            report += "- ������Ƶ��CAS����������δ��⵽�����쳣\n"
        
        # ���汨��
        report_path = os.path.join(output_dir, 'statistics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)


def get_args():
    """��ȡ�����в���"""
    parser = argparse.ArgumentParser(
        'Enhanced VideoMAE CAS evaluation with attention visualization',
        add_help=False)
    
    # ԭ�в���
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--device', default='cuda', help='device to use')
    
    # ���ӻ���ز���
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
    """������"""
    args = get_args()
    
    # ��ʼ���豸
    device = torch.device(args.device)
    
    # ����ģ��
    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        num_classes=400,  # ����ʵ���������
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
    )
    
    model.to(device)
    
    # ������ǿ��CAS������
    cas_scorer = EnhancedCASScorer(args, model, device)
    
    if args.video_path:
        # ����Ƶ����
        print(f"����������Ƶ: {args.video_path}")
        result = cas_scorer.evaluate_with_visualization(
            args.video_path, 
            save_visualization=args.enable_visualization
        )
        
        print(f"CAS����: {result['cas_score']:.3f}")
        if result['visualization']:
            print(f"���ӻ�����ѱ���")
            
    elif args.video_dir:
        # ��������
        print(f"����������ƵĿ¼: {args.video_dir}")
        results = cas_scorer.batch_evaluate_with_visualization(
            args.video_dir, 
            output_dir=args.output_dir
        )
        
        print(f"����������ɣ������� {len(results)} ����Ƶ")
        
    else:
        print("��ָ�� --video_path �� --video_dir ����")


if __name__ == '__main__':
    main()
