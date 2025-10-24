# -*- coding: utf-8 -*-
"""
CASע�������ӻ����Խű�
��ʾ���ʹ����ǿ��CAS����ϵͳ
"""

import os
import sys
import argparse
from pathlib import Path

# �����Ŀ·��
sys.path.insert(0, os.path.join(os.getcwd(), "VideoMAEv2"))

from enhanced_commonsense_adherence_score import EnhancedCASScorer, get_args
import torch


def test_single_video_visualization():
    """���Ե���Ƶע�������ӻ�"""
    
    print("=== ���Ե���Ƶע�������ӻ� ===")
    
    # ģ�����
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
    
    # ����ģ��ģ�ͣ�ʵ��ʹ��ʱ��Ҫ������ʵģ�ͣ�
    print("��ʼ��ģ��...")
    # model = create_model(args.model, ...)  # ʵ��ʹ��ʱȡ��ע��
    model = None  # ģ��ģ��
    
    if model is None:
        print("����: ʹ��ģ��ģ�ͽ�����ʾ")
        print("ʵ��ʹ��ʱ�������ʵ��VideoMAEv2ģ��")
        return
    
    # ����CAS������
    cas_scorer = EnhancedCASScorer(args, model, args.device)
    
    # ������Ƶ·�������滻Ϊʵ����Ƶ·����
    test_video = "path/to/your/test_video.mp4"
    
    if not os.path.exists(test_video):
        print(f"������Ƶ������: {test_video}")
        print("�뽫������Ƶ·���滻Ϊʵ�ʴ��ڵ���Ƶ�ļ�")
        return
    
    print(f"������Ƶ: {test_video}")
    
    try:
        # ִ������
        result = cas_scorer.evaluate_with_visualization(test_video)
        
        print(f"CAS����: {result['cas_score']:.3f}")
        
        if result['visualization']:
            print("? ������ע�������ӻ�")
            print(f"  ���Ŀ¼: {result['visualization']['output_dir']}")
        else:
            print("? CAS����������������ӻ�")
            
    except Exception as e:
        print(f"���������г���: {e}")


def test_batch_visualization():
    """����������Ƶ���ӻ�"""
    
    print("\n=== ����������Ƶ���ӻ� ===")
    
    # ģ�����
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
    
    # ����ģ��ģ��
    model = None  # ģ��ģ��
    
    if model is None:
        print("����: ʹ��ģ��ģ�ͽ�����ʾ")
        return
    
    # ����CAS������
    cas_scorer = EnhancedCASScorer(args, model, args.device)
    
    # ������ƵĿ¼�����滻Ϊʵ��Ŀ¼��
    test_video_dir = "path/to/your/video/directory"
    
    if not os.path.exists(test_video_dir):
        print(f"������ƵĿ¼������: {test_video_dir}")
        print("�뽫����Ŀ¼·���滻Ϊʵ�ʴ��ڵ�Ŀ¼")
        return
    
    print(f"����������ƵĿ¼: {test_video_dir}")
    
    try:
        # ִ����������
        results = cas_scorer.batch_evaluate_with_visualization(
            test_video_dir, 
            output_dir=args.output_dir
        )
        
        print(f"? �����������")
        print(f"  ������Ƶ����: {len(results)}")
        print(f"  �������Ŀ¼: {args.output_dir}")
        
        # ͳ����Ϣ
        scores = [r.get('cas_score', 0.0) for r in results if 'error' not in r]
        low_scores = [s for s in scores if s < 0.5]
        visualizations = [r for r in results if r.get('visualization') is not None]
        
        print(f"  ƽ��CAS����: {sum(scores)/len(scores):.3f}")
        print(f"  �ͷ���Ƶ����: {len(low_scores)}")
        print(f"  ���ɿ��ӻ�����: {len(visualizations)}")
        
    except Exception as e:
        print(f"�������������г���: {e}")


def demonstrate_attention_analysis():
    """��ʾע������������"""
    
    print("\n=== ע��������������ʾ ===")
    
    # ģ��ע����Ȩ������
    import numpy as np
    
    # ����ģ���ע��������ͼ
    attention_heatmap = np.random.rand(224, 224)
    
    # ģ���ע���������쳣����
    attention_heatmap[100:150, 100:150] += 0.5  # ����ĳ�������ע����
    
    print("ע�����������:")
    print(f"  ע����ͼ�ߴ�: {attention_heatmap.shape}")
    print(f"  ƽ��ע����: {np.mean(attention_heatmap):.3f}")
    print(f"  ���ע����: {np.max(attention_heatmap):.3f}")
    print(f"  ע������׼��: {np.std(attention_heatmap):.3f}")
    
    # �ҳ���ע��������
    threshold = np.percentile(attention_heatmap, 90)
    high_attention_regions = np.where(attention_heatmap > threshold)
    
    print(f"  ��ע������������: {len(high_attention_regions[0])}")
    print(f"  ��ע������ֵ: {threshold:.3f}")
    
    # ģ��CAS����
    cas_score = 0.3  # �ͷ�
    
    print(f"\nCAS���ַ���:")
    print(f"  CAS����: {cas_score:.3f}")
    
    if cas_score < 0.5:
        print("  ? ��⵽��CAS���֣����ܴ��ڳ�ʶΥ��")
        print("  ? �������ע�����������˽��쳣ԭ��")
    else:
        print("  ? CAS����������δ��⵽�����쳣")


def create_usage_example():
    """����ʹ��ʾ��"""
    
    example_code = '''
# CASע�������ӻ�ʹ��ʾ��

# 1. ����Ƶ����
from enhanced_commonsense_adherence_score import EnhancedCASScorer
import torch

# ��ʼ������
args = {
    'model': 'vit_base_patch16_224',
    'input_size': 224,
    'num_frames': 16,
    'tubelet_size': 2,
    'device': 'cuda',
    'enable_visualization': True,
    'visualization_threshold': 0.5
}

# ����ģ��
model = create_model(args['model'], ...)
model.to(args['device'])

# ����CAS������
cas_scorer = EnhancedCASScorer(args, model, args['device'])

# ����������Ƶ
result = cas_scorer.evaluate_with_visualization('path/to/video.mp4')
print(f"CAS����: {result['cas_score']:.3f}")

# 2. ��������
results = cas_scorer.batch_evaluate_with_visualization(
    'path/to/video/directory',
    output_dir='./visualization_results'
)

# 3. ������ʹ��
# python enhanced_commonsense_adherence_score.py \\
#     --video_path path/to/video.mp4 \\
#     --enable_visualization \\
#     --output_dir ./results

# python enhanced_commonsense_adherence_score.py \\
#     --video_dir path/to/videos \\
#     --enable_visualization \\
#     --output_dir ./batch_results
'''
    
    print("\n=== ʹ��ʾ�� ===")
    print(example_code)


def main():
    """������"""
    parser = argparse.ArgumentParser(description='CASע�������ӻ�����')
    parser.add_argument('--test_mode', choices=['single', 'batch', 'demo', 'example'], 
                       default='demo', help='����ģʽ')
    
    args = parser.parse_args()
    
    print("CASע����Ȩ�ؿ��ӻ�����")
    print("=" * 50)
    
    if args.test_mode == 'single':
        test_single_video_visualization()
    elif args.test_mode == 'batch':
        test_batch_visualization()
    elif args.test_mode == 'demo':
        demonstrate_attention_analysis()
    elif args.test_mode == 'example':
        create_usage_example()
    
    print("\n������ɣ�")
    print("\nע������:")
    print("1. ʵ��ʹ��ʱ��Ҫ������ʵ��VideoMAEv2ģ��")
    print("2. ��ȷ����Ƶ�ļ�·����ȷ")
    print("3. ���ӻ���������浽ָ�����Ŀ¼")
    print("4. ��CAS���ֵ���Ƶ���Զ�����ע�������ӻ�")


if __name__ == '__main__':
    main()
