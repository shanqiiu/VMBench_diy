# -*- coding: utf-8 -*-
"""
�򻯰���Ƶģ�������
����VMBench��MSS��������ר�����ڿ���ģ�����
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Tuple
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ���VMBench·��
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "Q-Align"))

# ����VMBench���ģ��
from motion_smoothness_score import QAlignVideoScorer, load_video_sliding_window, set_threshold, get_artifacts_frames


class SimpleBlurDetector:
    """�򻯰�ģ�����������Ҫ����MSS����"""
    
    def __init__(self, device="cuda:0", model_path=".cache/q-future/one-align"):
        """
        ��ʼ���򻯰�ģ�������
        
        Args:
            device: �����豸
            model_path: Q-Alignģ��·��
        """
        self.device = device
        self.model_path = model_path
        
        # ��ʼ��Q-Alignģ��
        print("���ڳ�ʼ��Q-Alignģ��...")
        try:
            self.q_align_scorer = QAlignVideoScorer(
                pretrained=model_path, 
                device=device
            )
            print("Q-Alignģ�ͳ�ʼ����ɣ�")
        except Exception as e:
            print(f"ģ�ͳ�ʼ��ʧ��: {e}")
            raise
        
        # ģ��������
        self.blur_thresholds = {
            'mild_blur': 0.015,    # ��΢ģ����ֵ
            'moderate_blur': 0.025, # �е�ģ����ֵ
            'severe_blur': 0.04    # ����ģ����ֵ
        }
    
    def detect_blur(self, video_path: str, window_size: int = 3) -> Dict:
        """
        �����Ƶģ��
        
        Args:
            video_path: ��Ƶ�ļ�·��
            window_size: �������ڴ�С
            
        Returns:
            ģ�������
        """
        print(f"��ʼ�����Ƶģ��: {os.path.basename(video_path)}")
        
        try:
            # 1. ������Ƶ��������������
            video_frames = load_video_sliding_window(video_path, window_size=window_size)
            _, _, quality_scores = self.q_align_scorer(video_frames)
            quality_scores = quality_scores.tolist()
            
            # 2. ��������˶�����
            camera_movement = self._estimate_camera_movement(video_path)
            
            # 3. ��������Ӧ��ֵ
            threshold = set_threshold(camera_movement)
            
            # 4. ���ģ��֡
            blur_frames = get_artifacts_frames(quality_scores, threshold)
            
            # 5. ����ģ��ָ��
            blur_metrics = self._calculate_blur_metrics(quality_scores, blur_frames, threshold)
            
            # 6. ���ɼ����
            result = self._generate_detection_result(video_path, blur_metrics)
            
            return result
            
        except Exception as e:
            print(f"ģ�����ʧ��: {e}")
            return {
                'video_path': video_path,
                'blur_detected': False,
                'confidence': 0.0,
                'error': str(e),
                'blur_severity': '���ʧ��'
            }
    
    def _estimate_camera_movement(self, video_path: str) -> float:
        """��������˶�����"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # ��ȡǰ10֡�����˶�
            frame_count = 0
            while frame_count < 10:
                ret, frame = cap.read()
                if not ret:
                    break
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray_frame)
                frame_count += 1
            
            cap.release()
            
            if len(frames) < 2:
                return 0.0
            
            # ����֡�����
            total_diff = 0.0
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i], frames[i-1])
                total_diff += np.mean(diff)
            
            # ��һ���˶�����
            movement = total_diff / (len(frames) - 1) / 255.0
            return min(1.0, movement)
            
        except Exception as e:
            print(f"����˶�����ʧ��: {e}")
            return 0.0
    
    def _calculate_blur_metrics(self, quality_scores: List[float], blur_frames: List[int], threshold: float) -> Dict:
        """����ģ��ָ��"""
        total_frames = len(quality_scores)
        blur_frame_count = len(blur_frames)
        
        # ����ָ��
        blur_ratio = blur_frame_count / total_frames if total_frames > 0 else 0
        avg_quality = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        
        # �������������仯
        quality_diffs = np.abs(np.diff(quality_scores))
        max_quality_drop = np.max(quality_diffs) if len(quality_diffs) > 0 else 0
        
        # ����ģ�����س̶�
        blur_severity = self._determine_blur_severity(blur_ratio, max_quality_drop, threshold)
        
        # �����ۺ����Ŷ�
        confidence = self._calculate_confidence(blur_ratio, max_quality_drop, avg_quality)
        
        return {
            'total_frames': total_frames,
            'blur_frames': blur_frames,
            'blur_frame_count': blur_frame_count,
            'blur_ratio': blur_ratio,
            'avg_quality': avg_quality,
            'quality_std': quality_std,
            'max_quality_drop': max_quality_drop,
            'threshold': threshold,
            'blur_severity': blur_severity,
            'confidence': confidence,
            'blur_detected': blur_ratio > 0.05 or max_quality_drop > threshold
        }
    
    def _determine_blur_severity(self, blur_ratio: float, max_quality_drop: float, threshold: float) -> str:
        """ȷ��ģ�����س̶�"""
        if blur_ratio > 0.3 or max_quality_drop > threshold * 2:
            return "����ģ��"
        elif blur_ratio > 0.1 or max_quality_drop > threshold * 1.5:
            return "�е�ģ��"
        elif blur_ratio > 0.05 or max_quality_drop > threshold:
            return "��΢ģ��"
        else:
            return "��ģ��"
    
    def _calculate_confidence(self, blur_ratio: float, max_quality_drop: float, avg_quality: float) -> float:
        """����ģ��������Ŷ�"""
        # ����ģ�������������½��������Ŷ�
        blur_confidence = min(1.0, blur_ratio * 2)  # ģ������Ȩ��
        quality_confidence = min(1.0, max_quality_drop * 10)  # �����½�Ȩ��
        avg_quality_confidence = max(0.0, 1.0 - avg_quality)  # ƽ������Ȩ��
        
        # �ۺ����Ŷ�
        confidence = (blur_confidence * 0.4 + quality_confidence * 0.4 + avg_quality_confidence * 0.2)
        return min(1.0, confidence)
    
    def _generate_detection_result(self, video_path: str, blur_metrics: Dict) -> Dict:
        """���ɼ����"""
        return {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'blur_detected': blur_metrics['blur_detected'],
            'confidence': blur_metrics['confidence'],
            'blur_severity': blur_metrics['blur_severity'],
            'blur_ratio': blur_metrics['blur_ratio'],
            'blur_frame_count': blur_metrics['blur_frame_count'],
            'total_frames': blur_metrics['total_frames'],
            'avg_quality': blur_metrics['avg_quality'],
            'max_quality_drop': blur_metrics['max_quality_drop'],
            'threshold': blur_metrics['threshold'],
            'blur_frames': blur_metrics['blur_frames'],
            'recommendations': self._generate_recommendations(blur_metrics)
        }
    
    def _generate_recommendations(self, blur_metrics: Dict) -> List[str]:
        """���ɸĽ�����"""
        recommendations = []
        
        if blur_metrics['blur_detected']:
            severity = blur_metrics['blur_severity']
            
            if severity == "����ģ��":
                recommendations.append("��������¼����Ƶ")
                recommendations.append("ʹ�����żܻ��ȶ���")
                recommendations.append("�������Խ�����")
            elif severity == "�е�ģ��":
                recommendations.append("����ʹ���ȶ���")
                recommendations.append("���¼��֡��")
                recommendations.append("ȷ���������")
            else:  # ��΢ģ��
                recommendations.append("�ɿ��Ǻ��ڴ���")
                recommendations.append("��΢ģ����Ӱ���С")
        else:
            recommendations.append("��Ƶ��������")
            recommendations.append("�������⴦��")
        
        return recommendations
    
    def batch_detect(self, video_dir: str, output_dir: str = "./blur_detection_results") -> Dict:
        """���������Ƶģ��"""
        os.makedirs(output_dir, exist_ok=True)
        
        # ��ȡ������Ƶ�ļ�
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        results = []
        
        print(f"��ʼ������� {len(video_files)} ����Ƶ...")
        
        for video_file in video_files:
            try:
                result = self.detect_blur(str(video_file))
                results.append(result)
                print(f"  {video_file.name}: {result['blur_severity']} (���Ŷ�: {result['confidence']:.3f})")
                
            except Exception as e:
                print(f"  ���� {video_file.name} ʱ����: {e}")
                results.append({
                    'video_path': str(video_file),
                    'blur_detected': False,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # ������
        self._save_results(results, output_dir)
        
        return {
            'total_videos': len(video_files),
            'processed_videos': len(results),
            'blur_detected_count': sum(1 for r in results if r.get('blur_detected', False)),
            'results': results
        }
    
    def _save_results(self, results: List[Dict], output_dir: str):
        """��������"""
        # ����JSON���
        json_path = os.path.join(output_dir, 'blur_detection_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # ����CSVժҪ
        csv_path = os.path.join(output_dir, 'blur_detection_summary.csv')
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Video', 'Blur_Detected', 'Confidence', 'Severity', 'Blur_Ratio', 'Blur_Frames'])
            
            for result in results:
                writer.writerow([
                    os.path.basename(result.get('video_path', '')),
                    result.get('blur_detected', False),
                    f"{result.get('confidence', 0.0):.3f}",
                    result.get('blur_severity', ''),
                    f"{result.get('blur_ratio', 0.0):.3f}",
                    result.get('blur_frame_count', 0)
                ])
        
        # ����ͳ�Ʊ���
        self._generate_statistics_report(results, output_dir)
        
        print(f"������ѱ��浽: {output_dir}")
    
    def _generate_statistics_report(self, results: List[Dict], output_dir: str):
        """����ͳ�Ʊ���"""
        # ����ͳ����Ϣ
        total_videos = len(results)
        blur_detected_count = sum(1 for r in results if r.get('blur_detected', False))
        confidence_scores = [r.get('confidence', 0.0) for r in results if 'error' not in r]
        
        # ͳ��ģ�����س̶�
        severity_counts = {}
        for result in results:
            if result.get('blur_detected', False):
                severity = result.get('blur_severity', 'δ֪')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        report = f"""
# ��Ƶģ�����ͳ�Ʊ���

## ����ͳ��
- ����Ƶ����: {total_videos}
- ��⵽ģ������Ƶ: {blur_detected_count}
- ģ�������: {blur_detected_count/total_videos*100:.1f}%

## ���Ŷ�ͳ��
- ƽ�����Ŷ�: {np.mean(confidence_scores):.3f}
- ������Ŷ�: {np.min(confidence_scores):.3f}
- ������Ŷ�: {np.max(confidence_scores):.3f}
- ���Ŷȱ�׼��: {np.std(confidence_scores):.3f}

## ģ�����س̶ȷֲ�
"""
        
        for severity, count in severity_counts.items():
            report += f"- {severity}: {count} ����Ƶ\n"
        
        # ���汨��
        report_path = os.path.join(output_dir, 'statistics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    """������"""
    parser = argparse.ArgumentParser(description="�򻯰���Ƶģ�������")
    parser.add_argument("--video_path", type=str, help="������Ƶ�ļ�·��")
    parser.add_argument("--video_dir", type=str, help="��ƵĿ¼·��")
    parser.add_argument("--output_dir", type=str, default="./blur_detection_results", help="���Ŀ¼")
    parser.add_argument("--device", type=str, default="cuda:0", help="�����豸")
    parser.add_argument("--model_path", type=str, default=".cache/q-future/one-align", help="Q-Alignģ��·��")
    
    args = parser.parse_args()
    
    # ��ʼ�������
    detector = SimpleBlurDetector(device=args.device, model_path=args.model_path)
    
    if args.video_path:
        # ����Ƶ���
        print(f"��ⵥ����Ƶ: {args.video_path}")
        result = detector.detect_blur(args.video_path)
        
        print(f"\nģ�������:")
        print(f"  ��Ƶ: {result['video_name']}")
        print(f"  ��⵽ģ��: {result['blur_detected']}")
        print(f"  ���Ŷ�: {result['confidence']:.3f}")
        print(f"  ģ�����س̶�: {result['blur_severity']}")
        print(f"  ģ������: {result['blur_ratio']:.3f}")
        print(f"  ģ��֡��: {result['blur_frame_count']}/{result['total_frames']}")
        print(f"  ƽ������: {result['avg_quality']:.3f}")
        print(f"  ��������½�: {result['max_quality_drop']:.3f}")
        
        print(f"\n����:")
        for rec in result['recommendations']:
            print(f"  - {rec}")
        
        # ������
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, f"blur_detection_{os.path.basename(args.video_path)}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n����ѱ��浽: {result_path}")
        
    elif args.video_dir:
        # �������
        print(f"���������ƵĿ¼: {args.video_dir}")
        batch_results = detector.batch_detect(args.video_dir, args.output_dir)
        
        print(f"\n����������:")
        print(f"  ����Ƶ��: {batch_results['total_videos']}")
        print(f"  ����ɹ�: {batch_results['processed_videos']}")
        print(f"  ��⵽ģ��: {batch_results['blur_detected_count']}")
        
    else:
        print("��ָ�� --video_path �� --video_dir ����")


if __name__ == "__main__":
    main()
