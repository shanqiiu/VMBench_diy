# -*- coding: utf-8 -*-
"""
����VMBench����Ƶģ�����ϵͳ
��Ҫʹ��MSS (�˶�ƽ��������) �� PAS (�ɸ�֪��������) ����ģ�����
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ���VMBench·��
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "Q-Align"))

# ����VMBench���ģ��
from motion_smoothness_score import QAlignVideoScorer, load_video_sliding_window, set_threshold, get_artifacts_frames
from perceptible_amplitude_score import (
    load_video, load_model, get_grounding_output, 
    calculate_motion_degree, is_mask_suitable_for_tracking
)

# ����Co-Tracker
sys.path.append(os.path.join(os.getcwd(), "..", "co-tracker"))
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# ����Grounded-SAM
sys.path.append(os.path.join(os.getcwd(), "..", "Grounded-Segment-Anything"))
from segment_anything import sam_model_registry, SamPredictor


class BlurDetectionPipeline:
    """����VMBench����Ƶģ�����ܵ�"""
    
    def __init__(self, device="cuda:0", model_paths=None):
        """
        ��ʼ��ģ�����ܵ�
        
        Args:
            device: �����豸
            model_paths: ģ��·������
        """
        self.device = device
        self.model_paths = model_paths or self._get_default_model_paths()
        
        # ��ʼ��ģ��
        self._init_models()
        
        # ������
        self.blur_thresholds = {
            'mss_threshold': 0.025,  # MSS�����ֵ
            'pas_threshold': 0.1,   # PAS�����ֵ
            'confidence_threshold': 0.7  # �ۺ����Ŷ���ֵ
        }
        
    def _get_default_model_paths(self):
        """��ȡĬ��ģ��·��"""
        return {
            'q_align_model': ".cache/q-future/one-align",
            'grounding_dino_config': "../Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py",
            'grounding_dino_checkpoint': ".cache/groundingdino_swinb_cogcoor.pth",
            'bert_path': ".cache/google-bert/bert-base-uncased",
            'sam_checkpoint': ".cache/sam_vit_h_4b8939.pth",
            'cotracker_checkpoint': ".cache/scaled_offline.pth"
        }
    
    def _init_models(self):
        """��ʼ��������Ҫ��ģ��"""
        print("���ڳ�ʼ��ģ�����ģ��...")
        
        try:
            # ��ʼ��Q-Alignģ�� (MSS������)
            print("  ��ʼ��Q-Alignģ��...")
            self.q_align_scorer = QAlignVideoScorer(
                pretrained=self.model_paths['q_align_model'], 
                device=self.device
            )
            
            # ��ʼ��GroundingDINOģ��
            print("  ��ʼ��GroundingDINOģ��...")
            self.grounding_model = load_model(
                self.model_paths['grounding_dino_config'],
                self.model_paths['grounding_dino_checkpoint'],
                self.model_paths['bert_path'],
                device=self.device
            )
            
            # ��ʼ��SAMģ��
            print("  ��ʼ��SAMģ��...")
            sam_predictor = SamPredictor(
                sam_model_registry["vit_h"](checkpoint=self.model_paths['sam_checkpoint']).to(self.device)
            )
            self.sam_predictor = sam_predictor
            
            # ��ʼ��Co-Trackerģ��
            print("  ��ʼ��Co-Trackerģ��...")
            self.cotracker_model = CoTrackerPredictor(
                checkpoint=self.model_paths['cotracker_checkpoint'],
                v2=False,
                offline=True,
                window_len=60,
            ).to(self.device)
            
            print("����ģ�ͳ�ʼ����ɣ�")
            
        except Exception as e:
            print(f"ģ�ͳ�ʼ��ʧ��: {e}")
            raise
    
    def detect_blur_in_video(self, video_path: str, subject_noun: str = "person") -> Dict:
        """
        �����Ƶ�е�ģ���쳣
        
        Args:
            video_path: ��Ƶ�ļ�·��
            subject_noun: �����������
            
        Returns:
            ������ֵ�
        """
        print(f"��ʼ�����Ƶģ��: {video_path}")
        
        try:
            # 1. ʹ��MSS���������ģ��
            mss_results = self._detect_blur_with_mss(video_path)
            
            # 2. ʹ��PAS������������֤
            pas_results = self._detect_blur_with_pas(video_path, subject_noun)
            
            # 3. �ۺ��ж�ģ�������
            blur_results = self._combine_blur_detection(mss_results, pas_results)
            
            # 4. ���ɼ�ⱨ��
            detection_report = self._generate_blur_report(video_path, blur_results)
            
            return detection_report
            
        except Exception as e:
            print(f"ģ���������г���: {e}")
            return {
                'blur_detected': False,
                'confidence': 0.0,
                'error': str(e),
                'mss_score': 0.0,
                'pas_score': 0.0,
                'blur_frames': []
            }
    
    def _detect_blur_with_mss(self, video_path: str) -> Dict:
        """ʹ��MSS���������ģ��"""
        try:
            # ������Ƶ��������������
            video_frames = load_video_sliding_window(video_path, window_size=3)
            _, _, quality_scores = self.q_align_scorer(video_frames)
            quality_scores = quality_scores.tolist()
            
            # ��������˶����ȣ����ڵ�����ֵ��
            camera_movement = self._estimate_camera_movement(video_path)
            
            # ��������Ӧ��ֵ
            threshold = set_threshold(camera_movement)
            
            # ���ģ��֡
            blur_frames = get_artifacts_frames(quality_scores, threshold)
            
            # ����MSS����
            mss_score = 1 - len(blur_frames) / len(quality_scores)
            
            return {
                'mss_score': mss_score,
                'blur_frames': blur_frames.tolist(),
                'quality_scores': quality_scores,
                'threshold': threshold,
                'camera_movement': camera_movement
            }
            
        except Exception as e:
            print(f"MSS���ʧ��: {e}")
            return {
                'mss_score': 0.0,
                'blur_frames': [],
                'quality_scores': [],
                'threshold': 0.025,
                'camera_movement': 0.0,
                'error': str(e)
            }
    
    def _detect_blur_with_pas(self, video_path: str, subject_noun: str) -> Dict:
        """ʹ��PAS�������������ģ��"""
        try:
            # ������Ƶ
            image_pil, image, image_array, video = load_video(video_path)
            
            # ����������
            text_prompt = subject_noun + '.'
            boxes_filt, pred_phrases = get_grounding_output(
                self.grounding_model, image, text_prompt, 
                box_threshold=0.3, text_threshold=0.25, device=self.device
            )
            
            if boxes_filt.shape[0] == 0:
                return {
                    'pas_score': 0.0,
                    'subject_detected': False,
                    'motion_degree': 0.0,
                    'error': f"No {subject_noun} detected"
                }
            
            # ������������
            self.sam_predictor.set_image(image_array)
            boxes_filt = boxes_filt.cpu()
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_filt, image.shape[:2]
            ).to(self.device)
            
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            # �����˶�����
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
            video_width, video_height = video_tensor.shape[-1], video_tensor.shape[-2]
            video_tensor = video_tensor.to(self.device)
            
            # ������������
            subject_mask = torch.any(masks, dim=0).to(torch.uint8) * 255
            subject_mask = subject_mask.unsqueeze(0)
            
            # ��������Ƿ��ʺϸ���
            if not is_mask_suitable_for_tracking(subject_mask, video_width, video_height, 30):
                return {
                    'pas_score': 0.0,
                    'subject_detected': True,
                    'motion_degree': 0.0,
                    'error': "Subject mask too small for tracking"
                }
            
            # ʹ��Co-Tracker�����˶�
            pred_tracks, pred_visibility = self.cotracker_model(
                video_tensor,
                grid_size=30,
                grid_query_frame=0,
                backward_tracking=True,
                segm_mask=subject_mask
            )
            
            if pred_tracks.shape[2] == 0:
                motion_degree = 0.0
            else:
                motion_degree = calculate_motion_degree(pred_tracks, video_width, video_height).item()
            
            # ģ���ᵼ���˶����ٲ�׼ȷ���˶������쳣��
            pas_score = min(1.0, motion_degree * 10)  # ��һ����0-1
            
            return {
                'pas_score': pas_score,
                'subject_detected': True,
                'motion_degree': motion_degree,
                'error': None
            }
            
        except Exception as e:
            print(f"PAS���ʧ��: {e}")
            return {
                'pas_score': 0.0,
                'subject_detected': False,
                'motion_degree': 0.0,
                'error': str(e)
            }
    
    def _estimate_camera_movement(self, video_path: str) -> float:
        """��������˶�����"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # ��ȡ�ؼ�֡
            frame_count = 0
            while frame_count < 10:  # ֻȡǰ10֡����
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
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
    
    def _combine_blur_detection(self, mss_results: Dict, pas_results: Dict) -> Dict:
        """�ۺ�MSS��PAS����ж�ģ��"""
        mss_score = mss_results.get('mss_score', 0.0)
        pas_score = pas_results.get('pas_score', 0.0)
        blur_frames = mss_results.get('blur_frames', [])
        
        # �����ۺ����Ŷ�
        # MSSȨ��0.8��PASȨ��0.2
        confidence = mss_score * 0.8 + pas_score * 0.2
        
        # �ж��Ƿ��⵽ģ��
        blur_detected = (
            len(blur_frames) > 0 and 
            confidence < self.blur_thresholds['confidence_threshold']
        )
        
        return {
            'blur_detected': blur_detected,
            'confidence': confidence,
            'mss_score': mss_score,
            'pas_score': pas_score,
            'blur_frames': blur_frames,
            'blur_severity': self._calculate_blur_severity(blur_frames, confidence)
        }
    
    def _calculate_blur_severity(self, blur_frames: List[int], confidence: float) -> str:
        """����ģ�����س̶�"""
        blur_ratio = len(blur_frames) / 100  # ������֡��100
        
        if blur_ratio > 0.3 or confidence < 0.3:
            return "����ģ��"
        elif blur_ratio > 0.1 or confidence < 0.5:
            return "�е�ģ��"
        elif blur_ratio > 0.05 or confidence < 0.7:
            return "��΢ģ��"
        else:
            return "��ģ��"
    
    def _generate_blur_report(self, video_path: str, blur_results: Dict) -> Dict:
        """����ģ����ⱨ��"""
        report = {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'detection_timestamp': str(np.datetime64('now')),
            'blur_detected': blur_results['blur_detected'],
            'confidence': blur_results['confidence'],
            'blur_severity': blur_results['blur_severity'],
            'mss_score': blur_results['mss_score'],
            'pas_score': blur_results['pas_score'],
            'blur_frames': blur_results['blur_frames'],
            'total_blur_frames': len(blur_results['blur_frames']),
            'blur_ratio': len(blur_results['blur_frames']) / 100.0,  # ������֡��100
            'recommendations': self._generate_recommendations(blur_results)
        }
        
        return report
    
    def _generate_recommendations(self, blur_results: Dict) -> List[str]:
        """���ɸĽ�����"""
        recommendations = []
        
        if blur_results['blur_detected']:
            if blur_results['blur_severity'] == "����ģ��":
                recommendations.append("��������¼����Ƶ��ȷ������ȶ�")
                recommendations.append("�������Խ�����")
            elif blur_results['blur_severity'] == "�е�ģ��":
                recommendations.append("����ʹ�����żܻ��ȶ���")
                recommendations.append("���¼��֡��")
            else:
                recommendations.append("��΢ģ�����ɿ��Ǻ��ڴ���")
        else:
            recommendations.append("��Ƶ�������ã����账��")
        
        return recommendations
    
    def batch_detect_blur(self, video_dir: str, output_dir: str = "./blur_detection_results") -> Dict:
        """���������Ƶģ��"""
        os.makedirs(output_dir, exist_ok=True)
        
        # ��ȡ������Ƶ�ļ�
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        results = []
        
        print(f"��ʼ������� {len(video_files)} ����Ƶ...")
        
        for video_file in tqdm(video_files, desc="ģ��������"):
            try:
                result = self.detect_blur_in_video(str(video_file))
                results.append(result)
                
            except Exception as e:
                print(f"������Ƶ {video_file.name} ʱ����: {e}")
                results.append({
                    'video_path': str(video_file),
                    'blur_detected': False,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # �����������
        self._save_batch_results(results, output_dir)
        
        return {
            'total_videos': len(video_files),
            'processed_videos': len(results),
            'blur_detected_count': sum(1 for r in results if r.get('blur_detected', False)),
            'results': results
        }
    
    def _save_batch_results(self, results: List[Dict], output_dir: str):
        """�������������"""
        # ����JSON���
        json_path = os.path.join(output_dir, 'blur_detection_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # ����CSVժҪ
        csv_path = os.path.join(output_dir, 'blur_detection_summary.csv')
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Video', 'Blur_Detected', 'Confidence', 'Severity', 'MSS_Score', 'PAS_Score', 'Blur_Frames'])
            
            for result in results:
                writer.writerow([
                    os.path.basename(result.get('video_path', '')),
                    result.get('blur_detected', False),
                    f"{result.get('confidence', 0.0):.3f}",
                    result.get('blur_severity', ''),
                    f"{result.get('mss_score', 0.0):.3f}",
                    f"{result.get('pas_score', 0.0):.3f}",
                    len(result.get('blur_frames', []))
                ])
        
        # ����ͳ�Ʊ���
        self._generate_statistics_report(results, output_dir)
        
        print(f"����������ѱ��浽: {output_dir}")
    
    def _generate_statistics_report(self, results: List[Dict], output_dir: str):
        """����ͳ�Ʊ���"""
        # ����ͳ����Ϣ
        total_videos = len(results)
        blur_detected_count = sum(1 for r in results if r.get('blur_detected', False))
        confidence_scores = [r.get('confidence', 0.0) for r in results if 'error' not in r]
        
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
        
        # ͳ��ģ�����س̶�
        severity_counts = {}
        for result in results:
            if result.get('blur_detected', False):
                severity = result.get('blur_severity', 'δ֪')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            report += f"- {severity}: {count} ����Ƶ\n"
        
        # ���汨��
        report_path = os.path.join(output_dir, 'statistics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    """������"""
    parser = argparse.ArgumentParser(description="����VMBench����Ƶģ�����")
    parser.add_argument("--video_path", type=str, help="������Ƶ�ļ�·��")
    parser.add_argument("--video_dir", type=str, help="��ƵĿ¼·��")
    parser.add_argument("--output_dir", type=str, default="./blur_detection_results", help="���Ŀ¼")
    parser.add_argument("--device", type=str, default="cuda:0", help="�����豸")
    parser.add_argument("--subject_noun", type=str, default="person", help="�����������")
    
    args = parser.parse_args()
    
    # ��ʼ�����ܵ�
    detector = BlurDetectionPipeline(device=args.device)
    
    if args.video_path:
        # ����Ƶ���
        print(f"��ⵥ����Ƶ: {args.video_path}")
        result = detector.detect_blur_in_video(args.video_path, args.subject_noun)
        
        print(f"ģ�������:")
        print(f"  ��⵽ģ��: {result['blur_detected']}")
        print(f"  ���Ŷ�: {result['confidence']:.3f}")
        print(f"  ģ�����س̶�: {result['blur_severity']}")
        print(f"  MSS����: {result['mss_score']:.3f}")
        print(f"  PAS����: {result['pas_score']:.3f}")
        
        # ������
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, f"blur_detection_{os.path.basename(args.video_path)}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"����ѱ��浽: {result_path}")
        
    elif args.video_dir:
        # �������
        print(f"���������ƵĿ¼: {args.video_dir}")
        batch_results = detector.batch_detect_blur(args.video_dir, args.output_dir)
        
        print(f"����������:")
        print(f"  ����Ƶ��: {batch_results['total_videos']}")
        print(f"  ����ɹ�: {batch_results['processed_videos']}")
        print(f"  ��⵽ģ��: {batch_results['blur_detected_count']}")
        
    else:
        print("��ָ�� --video_path �� --video_dir ����")


if __name__ == "__main__":
    main()
