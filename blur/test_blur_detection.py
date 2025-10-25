# -*- coding: utf-8 -*-
"""
��Ƶģ�������Խű�
���ڲ��Լ��ϵͳ�Ĺ���
"""

import os
import sys
import json
import tempfile
import numpy as np
import cv2
from pathlib import Path

# ��ӵ�ǰĿ¼��·��
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_blur_detector import SimpleBlurDetector
from blur_visualization import BlurVisualization
from config import BlurDetectionConfig


def create_test_video(output_path: str, blur_frames: list = None, duration: int = 5, fps: int = 30):
    """
    ����������Ƶ
    
    Args:
        output_path: �����Ƶ·��
        blur_frames: ��Ҫ���ģ����֡����
        duration: ��Ƶʱ�����룩
        fps: ֡��
    """
    if blur_frames is None:
        blur_frames = []
    
    # ��Ƶ����
    width, height = 640, 480
    total_frames = duration * fps
    
    # ������Ƶд����
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # ��������֡
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # ���һЩ����
        cv2.rectangle(frame, (50, 50), (width-50, height-50), (0, 255, 0), 2)
        cv2.putText(frame, f'Frame {frame_idx}', (100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # ����˶�Ԫ��
        center_x = width // 2 + int(50 * np.sin(frame_idx * 0.1))
        center_y = height // 2 + int(30 * np.cos(frame_idx * 0.1))
        cv2.circle(frame, (center_x, center_y), 20, (0, 0, 255), -1)
        
        # ���ģ��Ч��
        if frame_idx in blur_frames:
            # Ӧ�ø�˹ģ��
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        
        out.write(frame)
    
    out.release()
    print(f"������Ƶ�Ѵ���: {output_path}")


def test_simple_detector():
    """���Լ򻯰�����"""
    print("=== ���Լ򻯰����� ===")
    
    # ������ʱĿ¼
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # ����������Ƶ
        normal_video = temp_path / "normal_video.mp4"
        blur_video = temp_path / "blur_video.mp4"
        
        # ������Ƶ����ģ����
        create_test_video(str(normal_video))
        
        # ģ����Ƶ����10-15֡ģ����
        create_test_video(str(blur_video), blur_frames=list(range(10, 16)))
        
        try:
            # ��ʼ���������ʹ��CPU����CUDA���⣩
            detector = SimpleBlurDetector(device="cpu")
            
            # ����������Ƶ
            print("���������Ƶ...")
            normal_result = detector.detect_blur(str(normal_video))
            print(f"������Ƶ�����: {normal_result['blur_severity']} (���Ŷ�: {normal_result['confidence']:.3f})")
            
            # ����ģ����Ƶ
            print("���ģ����Ƶ...")
            blur_result = detector.detect_blur(str(blur_video))
            print(f"ģ����Ƶ�����: {blur_result['blur_severity']} (���Ŷ�: {blur_result['confidence']:.3f})")
            
            # ��֤���
            assert normal_result['blur_detected'] == False, "������ƵӦ�ü��Ϊ��ģ��"
            assert blur_result['blur_detected'] == True, "ģ����ƵӦ�ü��Ϊ��ģ��"
            
            print("? �򻯰���������ͨ��")
            
        except Exception as e:
            print(f"? �򻯰���������ʧ��: {e}")
            return False
    
    return True


def test_visualization():
    """���Կ��ӻ�����"""
    print("=== ���Կ��ӻ����� ===")
    
    try:
        # ������������
        test_result = {
            'video_path': 'test_video.mp4',
            'video_name': 'test_video.mp4',
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
            'recommendations': ['����ʹ���ȶ���', '���¼��֡��']
        }
        
        # ������ʱĿ¼
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = BlurVisualization(temp_dir)
            
            # ���Լ�ⱨ������
            report_path = visualizer.create_detection_report(test_result)
            assert os.path.exists(report_path), "��ⱨ��Ӧ�ñ�����"
            
            # ��������������ӻ�
            batch_results = [test_result, test_result.copy()]
            batch_viz_path = visualizer.visualize_batch_results(batch_results)
            assert os.path.exists(batch_viz_path), "�������ӻ�Ӧ�ñ�����"
            
            print("? ���ӻ����ܲ���ͨ��")
            
    except Exception as e:
        print(f"? ���ӻ����ܲ���ʧ��: {e}")
        return False
    
    return True


def test_config():
    """�������ù���"""
    print("=== �������ù��� ===")
    
    try:
        # ����Ĭ������
        config = BlurDetectionConfig()
        assert config.get_detection_param('window_size') == 3, "Ĭ�ϴ��ڴ�СӦ����3"
        assert config.get_detection_param('confidence_threshold') == 0.7, "Ĭ�����Ŷ���ֵӦ����0.7"
        
        # �������ø���
        config.update_detection_param('window_size', 5)
        assert config.get_detection_param('window_size') == 5, "���ø���Ӧ����Ч"
        
        # ����Ԥ������
        from config import get_preset_config
        fast_config = get_preset_config('fast')
        assert fast_config.get_detection_param('window_size') == 2, "�������ô��ڴ�СӦ����2"
        
        print("? ���ù��ܲ���ͨ��")
        
    except Exception as e:
        print(f"? ���ù��ܲ���ʧ��: {e}")
        return False
    
    return True


def test_batch_detection():
    """�����������"""
    print("=== ����������� ===")
    
    # ������ʱĿ¼
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        videos_dir = temp_path / "videos"
        videos_dir.mkdir()
        
        # �������������Ƶ
        for i in range(3):
            video_path = videos_dir / f"test_video_{i}.mp4"
            if i == 0:
                # ��һ����Ƶ��ģ��
                create_test_video(str(video_path))
            else:
                # ������Ƶ��ģ��
                blur_frames = list(range(5, 10)) if i == 1 else list(range(15, 25))
                create_test_video(str(video_path), blur_frames=blur_frames)
        
        try:
            # ��ʼ�������
            detector = SimpleBlurDetector(device="cpu")
            
            # �������
            results = detector.batch_detect(str(videos_dir), str(temp_path / "results"))
            
            # ��֤���
            assert results['total_videos'] == 3, "Ӧ�ü��3����Ƶ"
            assert results['processed_videos'] == 3, "Ӧ�ô���3����Ƶ"
            assert results['blur_detected_count'] >= 2, "Ӧ�ü�⵽����2��ģ����Ƶ"
            
            print("? ����������ͨ��")
            
        except Exception as e:
            print(f"? ����������ʧ��: {e}")
            return False
    
    return True


def run_all_tests():
    """�������в���"""
    print("��ʼ������Ƶģ�����ϵͳ����...")
    
    tests = [
        ("�򻯰�����", test_simple_detector),
        ("���ӻ�����", test_visualization),
        ("���ù���", test_config),
        ("�������", test_batch_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"? {test_name} ����ͨ��")
            else:
                print(f"? {test_name} ����ʧ��")
        except Exception as e:
            print(f"? {test_name} �����쳣: {e}")
    
    print(f"\n=== �����ܽ� ===")
    print(f"ͨ��: {passed}/{total}")
    print(f"�ɹ���: {passed/total*100:.1f}%")
    
    if passed == total:
        print("? ���в���ͨ����")
    else:
        print("?? ���ֲ���ʧ�ܣ�����ϵͳ����")


if __name__ == "__main__":
    run_all_tests()
