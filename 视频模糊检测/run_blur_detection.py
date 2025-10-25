# -*- coding: utf-8 -*-
"""
��Ƶģ��������нű�
�ṩ������ģ���������
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
import time

# ��ӵ�ǰĿ¼��·��
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_blur_detector import SimpleBlurDetector
from blur_detection_pipeline import BlurDetectionPipeline
from blur_visualization import BlurVisualization
from config import BlurDetectionConfig, get_preset_config


class BlurDetectionRunner:
    """ģ�����������"""
    
    def __init__(self, config: BlurDetectionConfig = None):
        """
        ��ʼ��������
        
        Args:
            config: �������
        """
        self.config = config or BlurDetectionConfig()
        self.detector = None
        self.visualizer = None
        
    def initialize_detector(self, use_simple: bool = True):
        """��ʼ�������"""
        print("���ڳ�ʼ�������...")
        
        if use_simple:
            # ʹ�ü򻯰�����
            self.detector = SimpleBlurDetector(
                device=self.config.get_device_config('device'),
                model_path=self.config.get_model_path('q_align_model')
            )
        else:
            # ʹ������������
            self.detector = BlurDetectionPipeline(
                device=self.config.get_device_config('device'),
                model_paths=self.config.model_paths
            )
        
        # ��ʼ�����ӻ�����
        self.visualizer = BlurVisualization(str(self.config.output_dir / "visualizations"))
        
        print("�������ʼ����ɣ�")
    
    def detect_single_video(self, video_path: str, generate_visualization: bool = True) -> Dict:
        """
        ��ⵥ����Ƶ
        
        Args:
            video_path: ��Ƶ·��
            generate_visualization: �Ƿ����ɿ��ӻ�
            
        Returns:
            �����
        """
        if not self.detector:
            self.initialize_detector()
        
        print(f"��ʼ�����Ƶ: {video_path}")
        start_time = time.time()
        
        # ִ�м��
        if hasattr(self.detector, 'detect_blur'):
            # �򻯰�����
            result = self.detector.detect_blur(video_path)
        else:
            # ����������
            result = self.detector.detect_blur_in_video(video_path)
        
        detection_time = time.time() - start_time
        result['detection_time'] = detection_time
        
        print(f"�����ɣ���ʱ: {detection_time:.2f}��")
        print(f"�����: {result.get('blur_severity', 'δ֪')} (���Ŷ�: {result.get('confidence', 0.0):.3f})")
        
        # ���ɿ��ӻ�
        if generate_visualization and self.visualizer:
            try:
                print("���ɿ��ӻ����...")
                if 'quality_scores' in result and 'blur_frames' in result:
                    # ���������������ӻ�
                    quality_viz_path = self.visualizer.visualize_quality_scores(
                        video_path, 
                        result['quality_scores'], 
                        result['blur_frames'], 
                        result.get('threshold', 0.025)
                    )
                    print(f"�����������ӻ��ѱ��浽: {quality_viz_path}")
                
                # ���ɼ�ⱨ��
                report_path = self.visualizer.create_detection_report(result)
                print(f"��ⱨ���ѱ��浽: {report_path}")
                
            except Exception as e:
                print(f"���ӻ�����ʧ��: {e}")
        
        return result
    
    def detect_batch_videos(self, video_dir: str, generate_visualization: bool = True) -> Dict:
        """
        ���������Ƶ
        
        Args:
            video_dir: ��ƵĿ¼
            generate_visualization: �Ƿ����ɿ��ӻ�
            
        Returns:
            ���������
        """
        if not self.detector:
            self.initialize_detector()
        
        print(f"��ʼ���������ƵĿ¼: {video_dir}")
        start_time = time.time()
        
        # ִ���������
        if hasattr(self.detector, 'batch_detect'):
            # �򻯰�����
            results = self.detector.batch_detect(video_dir, str(self.config.output_dir))
        else:
            # ����������
            results = self.detector.batch_detect_blur(video_dir, str(self.config.output_dir))
        
        detection_time = time.time() - start_time
        
        print(f"���������ɣ���ʱ: {detection_time:.2f}��")
        print(f"����Ƶ��: {results.get('total_videos', 0)}")
        print(f"��⵽ģ��: {results.get('blur_detected_count', 0)}")
        
        # �����������ӻ�
        if generate_visualization and self.visualizer and 'results' in results:
            try:
                print("�����������ӻ����...")
                batch_viz_path = self.visualizer.visualize_batch_results(results['results'])
                print(f"����������ӻ��ѱ��浽: {batch_viz_path}")
                
            except Exception as e:
                print(f"�������ӻ�����ʧ��: {e}")
        
        return results
    
    def save_results(self, results: Dict, filename: str = None):
        """��������"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"blur_detection_results_{timestamp}.json"
        
        output_path = self.config.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"������ѱ��浽: {output_path}")
        return str(output_path)


def main():
    """������"""
    parser = argparse.ArgumentParser(description="��Ƶģ��������нű�")
    parser.add_argument("--video_path", type=str, help="������Ƶ�ļ�·��")
    parser.add_argument("--video_dir", type=str, help="��ƵĿ¼·��")
    parser.add_argument("--output_dir", type=str, default="./blur_detection_results", help="���Ŀ¼")
    parser.add_argument("--config_preset", type=str, choices=['fast', 'accurate', 'balanced'], 
                       default='balanced', help="����Ԥ��")
    parser.add_argument("--use_simple", action='store_true', default=True, help="ʹ�ü򻯰�����")
    parser.add_argument("--no_visualization", action='store_true', help="�����ɿ��ӻ����")
    parser.add_argument("--device", type=str, default="cuda:0", help="�����豸")
    
    args = parser.parse_args()
    
    # ��������
    config = get_preset_config(args.config_preset)
    config.output_dir = Path(args.output_dir)
    config.update_device_config('device', args.device)
    
    # ��֤����
    if not config.validate_config():
        print("������֤ʧ�ܣ�����ģ���ļ���·������")
        return
    
    # ����������
    runner = BlurDetectionRunner(config)
    
    try:
        if args.video_path:
            # ����Ƶ���
            print("=== ����Ƶģ����� ===")
            result = runner.detect_single_video(
                args.video_path, 
                generate_visualization=not args.no_visualization
            )
            
            # ������
            runner.save_results(result)
            
        elif args.video_dir:
            # �������
            print("=== ������Ƶģ����� ===")
            results = runner.detect_batch_videos(
                args.video_dir, 
                generate_visualization=not args.no_visualization
            )
            
            # ������
            runner.save_results(results)
            
        else:
            print("��ָ�� --video_path �� --video_dir ����")
            return
        
        print("�����ɣ�")
        
    except KeyboardInterrupt:
        print("\n��ⱻ�û��ж�")
    except Exception as e:
        print(f"�������г��ִ���: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
