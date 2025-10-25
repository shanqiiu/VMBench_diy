# -*- coding: utf-8 -*-
"""
��Ƶģ�����ʹ��ʾ��
չʾ���ʹ��ģ�����ϵͳ
"""

import os
import sys
from pathlib import Path

# ��ӵ�ǰĿ¼��·��
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_blur_detector import SimpleBlurDetector
from blur_visualization import BlurVisualization
from config import BlurDetectionConfig, get_preset_config


def example_single_video_detection():
    """����Ƶ���ʾ��"""
    print("=== ����Ƶ���ʾ�� ===")
    
    # ������һ����Ƶ�ļ�
    video_path = "example_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"��Ƶ�ļ�������: {video_path}")
        print("�뽫������Ƶ�ļ�������Ϊ 'example_video.mp4' �����ڵ�ǰĿ¼")
        return
    
    try:
        # 1. ��ʼ�������
        print("��ʼ�������...")
        detector = SimpleBlurDetector(device="cuda:0")  # �����GPU
        
        # 2. ִ�м��
        print("��ʼ���...")
        result = detector.detect_blur(video_path)
        
        # 3. ��ʾ���
        print(f"\n�����:")
        print(f"  ��Ƶ: {result['video_name']}")
        print(f"  ��⵽ģ��: {result['blur_detected']}")
        print(f"  ���Ŷ�: {result['confidence']:.3f}")
        print(f"  ģ�����س̶�: {result['blur_severity']}")
        print(f"  ģ������: {result['blur_ratio']:.3f}")
        print(f"  ģ��֡��: {result['blur_frame_count']}/{result['total_frames']}")
        
        # 4. ��ʾ����
        print(f"\n�Ľ�����:")
        for rec in result['recommendations']:
            print(f"  ? {rec}")
        
        # 5. ���ɿ��ӻ�
        print("\n���ɿ��ӻ����...")
        visualizer = BlurVisualization("./visualization_results")
        
        # ���ɼ�ⱨ��
        report_path = visualizer.create_detection_report(result)
        print(f"��ⱨ���ѱ��浽: {report_path}")
        
    except Exception as e:
        print(f"�������г��ִ���: {e}")
        print("����ģ���ļ��Ƿ���ȷ��װ")


def example_batch_detection():
    """�������ʾ��"""
    print("\n=== �������ʾ�� ===")
    
    # ������һ����ƵĿ¼
    video_dir = "example_videos"
    
    if not os.path.exists(video_dir):
        print(f"��ƵĿ¼������: {video_dir}")
        print("�봴�� 'example_videos' Ŀ¼������һЩ��Ƶ�ļ�")
        return
    
    try:
        # 1. ��ʼ�������
        print("��ʼ�������...")
        detector = SimpleBlurDetector(device="cuda:0")
        
        # 2. ִ���������
        print("��ʼ�������...")
        results = detector.batch_detect(video_dir, "./batch_results")
        
        # 3. ��ʾͳ�ƽ��
        print(f"\n���������:")
        print(f"  ����Ƶ��: {results['total_videos']}")
        print(f"  ����ɹ�: {results['processed_videos']}")
        print(f"  ��⵽ģ��: {results['blur_detected_count']}")
        
        # 4. ��ʾÿ����Ƶ�Ľ��
        print(f"\n����Ƶ�����:")
        for result in results['results']:
            if 'error' not in result:
                print(f"  {os.path.basename(result['video_path'])}: {result['blur_severity']} (���Ŷ�: {result['confidence']:.3f})")
            else:
                print(f"  {os.path.basename(result['video_path'])}: ����ʧ�� - {result['error']}")
        
        # 5. �����������ӻ�
        print("\n�����������ӻ�...")
        visualizer = BlurVisualization("./batch_visualization")
        batch_viz_path = visualizer.visualize_batch_results(results['results'])
        print(f"�������ӻ��ѱ��浽: {batch_viz_path}")
        
    except Exception as e:
        print(f"�����������г��ִ���: {e}")


def example_custom_config():
    """�Զ�������ʾ��"""
    print("\n=== �Զ�������ʾ�� ===")
    
    try:
        # 1. �����Զ�������
        config = BlurDetectionConfig()
        
        # 2. �޸ļ�����
        config.update_detection_param('window_size', 5)  # ���󴰿�
        config.update_detection_param('confidence_threshold', 0.8)  # �����ֵ
        
        # 3. �޸��豸����
        config.update_device_config('device', 'cuda:0')
        
        print("�Զ�������:")
        print(f"  ���ڴ�С: {config.get_detection_param('window_size')}")
        print(f"  ���Ŷ���ֵ: {config.get_detection_param('confidence_threshold')}")
        print(f"  �����豸: {config.get_device_config('device')}")
        
        # 4. ʹ��Ԥ������
        fast_config = get_preset_config('fast')
        print(f"\n��������:")
        print(f"  ���ڴ�С: {fast_config.get_detection_param('window_size')}")
        print(f"  ���Ŷ���ֵ: {fast_config.get_detection_param('confidence_threshold')}")
        
    except Exception as e:
        print(f"����ʾ�������г��ִ���: {e}")


def example_visualization():
    """���ӻ�ʾ��"""
    print("\n=== ���ӻ�ʾ�� ===")
    
    try:
        # ����ʾ������
        example_result = {
            'video_path': 'example_video.mp4',
            'video_name': 'example_video.mp4',
            'blur_detected': True,
            'confidence': 0.75,
            'blur_severity': '�е�ģ��',
            'blur_ratio': 0.15,
            'blur_frame_count': 15,
            'total_frames': 100,
            'avg_quality': 0.75,
            'max_quality_drop': 0.12,
            'threshold': 0.025,
            'blur_frames': [10, 15, 20, 25, 30],
            'quality_scores': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3] * 10,
            'recommendations': ['����ʹ���ȶ���', '���¼��֡��', 'ȷ���������']
        }
        
        # ��ʼ�����ӻ�����
        visualizer = BlurVisualization("./example_visualization")
        
        # 1. ���ɼ�ⱨ��
        print("���ɼ�ⱨ��...")
        report_path = visualizer.create_detection_report(example_result)
        print(f"��ⱨ��: {report_path}")
        
        # 2. ���������������ӻ�
        print("���������������ӻ�...")
        quality_viz_path = visualizer.visualize_quality_scores(
            example_result['video_path'],
            example_result['quality_scores'],
            example_result['blur_frames'],
            example_result['threshold']
        )
        print(f"�����������ӻ�: {quality_viz_path}")
        
        # 3. ��������������ӻ�
        print("��������������ӻ�...")
        batch_results = [example_result, example_result.copy()]
        batch_viz_path = visualizer.visualize_batch_results(batch_results)
        print(f"����������ӻ�: {batch_viz_path}")
        
    except Exception as e:
        print(f"���ӻ�ʾ�������г��ִ���: {e}")


def main():
    """������"""
    print("��Ƶģ�����ϵͳʹ��ʾ��")
    print("=" * 50)
    
    # ���и���ʾ��
    example_single_video_detection()
    example_batch_detection()
    example_custom_config()
    example_visualization()
    
    print("\n" + "=" * 50)
    print("ʾ��������ɣ�")
    print("\nҪ����ʵ�ʵļ�⣬��:")
    print("1. ׼��������Ƶ�ļ�")
    print("2. ����: python run_blur_detection.py --video_path your_video.mp4")
    print("3. ������: python run_blur_detection.py --video_dir your_video_directory")


if __name__ == "__main__":
    main()
