# -*- coding: utf-8 -*-
"""
基于VMBench的视频模糊检测系统
主要使用MSS (运动平滑度评分) 和 PAS (可感知幅度评分) 进行模糊检测
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

# 添加VMBench路径
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (blur目录的父目录)
project_root = os.path.dirname(current_dir)

# 保存原始工作目录
original_cwd = os.getcwd()

try:
    # 临时切换到项目根目录以便导入
    os.chdir(project_root)
    
    # 添加项目根目录到路径
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "Q-Align"))
    sys.path.append(os.path.join(project_root, "Grounded-Segment-Anything"))
    sys.path.append(os.path.join(project_root, "Grounded-Segment-Anything", "GroundingDINO"))
    sys.path.append(os.path.join(project_root, "Grounded-Segment-Anything", "segment_anything"))
    sys.path.append(os.path.join(project_root, "co-tracker"))
    
    # 导入VMBench相关模块
    from motion_smoothness_score import QAlignVideoScorer, load_video_sliding_window, set_threshold, get_artifacts_frames
    from perceptible_amplitude_score import (
        load_video, load_model, get_grounding_output, 
        calculate_motion_degree, is_mask_suitable_for_tracking
    )
    
    # 导入Co-Tracker
    from cotracker.utils.visualizer import Visualizer, read_video_from_path
    from cotracker.predictor import CoTrackerPredictor
    
    # 导入Grounded-SAM
    from segment_anything import sam_model_registry, SamPredictor
    
finally:
    # 恢复原始工作目录
    os.chdir(original_cwd)


class BlurDetectionPipeline:
    """基于VMBench的视频模糊检测管道"""
    
    def __init__(self, device="cuda:0", model_paths=None):
        """
        初始化模糊检测管道
        
        Args:
            device: 计算设备
            model_paths: 模型路径配置
        """
        self.device = device
        self.model_paths = model_paths or self._get_default_model_paths()
        
        # 初始化模型
        self._init_models()
        
        # 检测参数
        self.blur_thresholds = {
            'mss_threshold': 0.025,  # MSS检测阈值
            'pas_threshold': 0.1,   # PAS检测阈值
            'confidence_threshold': 0.7  # 综合置信度阈值
        }
        
    def _get_default_model_paths(self):
        """获取默认模型路径"""
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        return {
            'q_align_model': ".cache/q-future/one-align",
            'grounding_dino_config': os.path.join(project_root, "Grounded-Segment-Anything", "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinB.py"),
            'grounding_dino_checkpoint': ".cache/groundingdino_swinb_cogcoor.pth",
            'bert_path': ".cache/google-bert/bert-base-uncased",
            'sam_checkpoint': ".cache/sam_vit_h_4b8939.pth",
            'cotracker_checkpoint': ".cache/scaled_offline.pth"
        }
    
    def _init_models(self):
        """初始化所有需要的模型"""
        print("正在初始化模糊检测模型...")
        
        try:
            # 初始化Q-Align模型 (MSS评分器)
            print("  初始化Q-Align模型...")
            self.q_align_scorer = QAlignVideoScorer(
                pretrained=self.model_paths['q_align_model'], 
                device=self.device
            )
            
            # 初始化GroundingDINO模型
            print("  初始化GroundingDINO模型...")
            self.grounding_model = load_model(
                self.model_paths['grounding_dino_config'],
                self.model_paths['grounding_dino_checkpoint'],
                self.model_paths['bert_path'],
                device=self.device
            )
            
            # 初始化SAM模型
            print("  初始化SAM模型...")
            sam_predictor = SamPredictor(
                sam_model_registry["vit_h"](checkpoint=self.model_paths['sam_checkpoint']).to(self.device)
            )
            self.sam_predictor = sam_predictor
            
            # 初始化Co-Tracker模型
            print("  初始化Co-Tracker模型...")
            self.cotracker_model = CoTrackerPredictor(
                checkpoint=self.model_paths['cotracker_checkpoint'],
                v2=False,
                offline=True,
                window_len=60,
            ).to(self.device)
            
            print("所有模型初始化完成！")
            
        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise
    
    def detect_blur_in_video(self, video_path: str, subject_noun: str = "person") -> Dict:
        """
        检测视频中的模糊异常
        
        Args:
            video_path: 视频文件路径
            subject_noun: 主体对象名称
            
        Returns:
            检测结果字典
        """
        print(f"开始检测视频模糊: {video_path}")
        
        try:
            # 1. 使用MSS评分器检测模糊
            mss_results = self._detect_blur_with_mss(video_path)
            
            # 2. 使用PAS评分器辅助验证
            pas_results = self._detect_blur_with_pas(video_path, subject_noun)
            
            # 3. 综合判断模糊检测结果
            blur_results = self._combine_blur_detection(mss_results, pas_results)
            
            # 4. 生成检测报告
            detection_report = self._generate_blur_report(video_path, blur_results)
            
            return detection_report
            
        except Exception as e:
            print(f"模糊检测过程中出错: {e}")
            return {
                'blur_detected': False,
                'confidence': 0.0,
                'error': str(e),
                'mss_score': 0.0,
                'pas_score': 0.0,
                'blur_frames': []
            }
    
    def _detect_blur_with_mss(self, video_path: str) -> Dict:
        """使用MSS评分器检测模糊"""
        try:
            # 加载视频并计算质量分数
            video_frames = load_video_sliding_window(video_path, window_size=3)
            _, _, quality_scores = self.q_align_scorer(video_frames)
            quality_scores = quality_scores.tolist()
            
            # 计算相机运动幅度（用于调整阈值）
            camera_movement = self._estimate_camera_movement(video_path)
            
            # 设置自适应阈值
            threshold = set_threshold(camera_movement)
            
            # 检测模糊帧
            blur_frames = get_artifacts_frames(quality_scores, threshold)
            
            # 计算MSS分数
            mss_score = 1 - len(blur_frames) / len(quality_scores)
            
            return {
                'mss_score': mss_score,
                'blur_frames': blur_frames.tolist(),
                'quality_scores': quality_scores,
                'threshold': threshold,
                'camera_movement': camera_movement
            }
            
        except Exception as e:
            print(f"MSS检测失败: {e}")
            return {
                'mss_score': 0.0,
                'blur_frames': [],
                'quality_scores': [],
                'threshold': 0.025,
                'camera_movement': 0.0,
                'error': str(e)
            }
    
    def _detect_blur_with_pas(self, video_path: str, subject_noun: str) -> Dict:
        """使用PAS评分器辅助检测模糊"""
        try:
            # 加载视频
            image_pil, image, image_array, video = load_video(video_path)
            
            # 检测主体对象
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
            
            # 生成主体掩码
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
            
            # 计算运动幅度
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
            video_width, video_height = video_tensor.shape[-1], video_tensor.shape[-2]
            video_tensor = video_tensor.to(self.device)
            
            # 创建主体掩码
            subject_mask = torch.any(masks, dim=0).to(torch.uint8) * 255
            subject_mask = subject_mask.unsqueeze(0)
            
            # 检查掩码是否适合跟踪
            if not is_mask_suitable_for_tracking(subject_mask, video_width, video_height, 30):
                return {
                    'pas_score': 0.0,
                    'subject_detected': True,
                    'motion_degree': 0.0,
                    'error': "Subject mask too small for tracking"
                }
            
            # 使用Co-Tracker跟踪运动
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
            
            # 模糊会导致运动跟踪不准确，运动幅度异常低
            pas_score = min(1.0, motion_degree * 10)  # 归一化到0-1
            
            return {
                'pas_score': pas_score,
                'subject_detected': True,
                'motion_degree': motion_degree,
                'error': None
            }
            
        except Exception as e:
            print(f"PAS检测失败: {e}")
            return {
                'pas_score': 0.0,
                'subject_detected': False,
                'motion_degree': 0.0,
                'error': str(e)
            }
    
    def _estimate_camera_movement(self, video_path: str) -> float:
        """估算相机运动幅度"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # 读取关键帧
            frame_count = 0
            while frame_count < 10:  # 只取前10帧估算
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                frame_count += 1
            
            cap.release()
            
            if len(frames) < 2:
                return 0.0
            
            # 计算帧间差异
            total_diff = 0.0
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i], frames[i-1])
                total_diff += np.mean(diff)
            
            # 归一化运动幅度
            movement = total_diff / (len(frames) - 1) / 255.0
            return min(1.0, movement)
            
        except Exception as e:
            print(f"相机运动估算失败: {e}")
            return 0.0
    
    def _combine_blur_detection(self, mss_results: Dict, pas_results: Dict) -> Dict:
        """综合MSS和PAS结果判断模糊"""
        mss_score = mss_results.get('mss_score', 0.0)
        pas_score = pas_results.get('pas_score', 0.0)
        blur_frames = mss_results.get('blur_frames', [])
        
        # 计算综合置信度
        # MSS权重0.8，PAS权重0.2
        confidence = mss_score * 0.8 + pas_score * 0.2
        
        # 判断是否检测到模糊
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
        """计算模糊严重程度"""
        blur_ratio = len(blur_frames) / 100  # 假设总帧数100
        
        if blur_ratio > 0.3 or confidence < 0.3:
            return "严重模糊"
        elif blur_ratio > 0.1 or confidence < 0.5:
            return "中等模糊"
        elif blur_ratio > 0.05 or confidence < 0.7:
            return "轻微模糊"
        else:
            return "无模糊"
    
    def _generate_blur_report(self, video_path: str, blur_results: Dict) -> Dict:
        """生成模糊检测报告"""
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
            'blur_ratio': len(blur_results['blur_frames']) / 100.0,  # 假设总帧数100
            'recommendations': self._generate_recommendations(blur_results)
        }
        
        return report
    
    def _generate_recommendations(self, blur_results: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if blur_results['blur_detected']:
            if blur_results['blur_severity'] == "严重模糊":
                recommendations.append("建议重新录制视频，确保相机稳定")
                recommendations.append("检查相机对焦设置")
            elif blur_results['blur_severity'] == "中等模糊":
                recommendations.append("建议使用三脚架或稳定器")
                recommendations.append("提高录制帧率")
            else:
                recommendations.append("轻微模糊，可考虑后期处理")
        else:
            recommendations.append("视频质量良好，无需处理")
        
        return recommendations
    
    def batch_detect_blur(self, video_dir: str, output_dir: str = "./blur_detection_results") -> Dict:
        """批量检测视频模糊"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有视频文件
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        results = []
        
        print(f"开始批量检测 {len(video_files)} 个视频...")
        
        for video_file in tqdm(video_files, desc="模糊检测进度"):
            try:
                result = self.detect_blur_in_video(str(video_file))
                results.append(result)
                
            except Exception as e:
                print(f"处理视频 {video_file.name} 时出错: {e}")
                results.append({
                    'video_path': str(video_file),
                    'blur_detected': False,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # 保存批量结果
        self._save_batch_results(results, output_dir)
        
        return {
            'total_videos': len(video_files),
            'processed_videos': len(results),
            'blur_detected_count': sum(1 for r in results if r.get('blur_detected', False)),
            'results': results
        }
    
    def _save_batch_results(self, results: List[Dict], output_dir: str):
        """保存批量检测结果"""
        # 保存JSON结果
        json_path = os.path.join(output_dir, 'blur_detection_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV摘要
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
        
        # 生成统计报告
        self._generate_statistics_report(results, output_dir)
        
        print(f"批量检测结果已保存到: {output_dir}")
    
    def _generate_statistics_report(self, results: List[Dict], output_dir: str):
        """生成统计报告"""
        # 计算统计信息
        total_videos = len(results)
        blur_detected_count = sum(1 for r in results if r.get('blur_detected', False))
        confidence_scores = [r.get('confidence', 0.0) for r in results if 'error' not in r]
        
        report = f"""
# 视频模糊检测统计报告

## 基本统计
- 总视频数量: {total_videos}
- 检测到模糊的视频: {blur_detected_count}
- 模糊检测率: {blur_detected_count/total_videos*100:.1f}%

## 置信度统计
- 平均置信度: {np.mean(confidence_scores):.3f}
- 最低置信度: {np.min(confidence_scores):.3f}
- 最高置信度: {np.max(confidence_scores):.3f}
- 置信度标准差: {np.std(confidence_scores):.3f}

## 模糊严重程度分布
"""
        
        # 统计模糊严重程度
        severity_counts = {}
        for result in results:
            if result.get('blur_detected', False):
                severity = result.get('blur_severity', '未知')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            report += f"- {severity}: {count} 个视频\n"
        
        # 保存报告
        report_path = os.path.join(output_dir, 'statistics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于VMBench的视频模糊检测")
    parser.add_argument("--video_path", type=str, help="单个视频文件路径")
    parser.add_argument("--video_dir", type=str, help="视频目录路径")
    parser.add_argument("--output_dir", type=str, default="./blur_detection_results", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    parser.add_argument("--subject_noun", type=str, default="person", help="主体对象名称")
    
    args = parser.parse_args()
    
    # 初始化检测管道
    detector = BlurDetectionPipeline(device=args.device)
    
    if args.video_path:
        # 单视频检测
        print(f"检测单个视频: {args.video_path}")
        result = detector.detect_blur_in_video(args.video_path, args.subject_noun)
        
        print(f"模糊检测结果:")
        print(f"  检测到模糊: {result['blur_detected']}")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  模糊严重程度: {result['blur_severity']}")
        print(f"  MSS评分: {result['mss_score']:.3f}")
        print(f"  PAS评分: {result['pas_score']:.3f}")
        
        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, f"blur_detection_{os.path.basename(args.video_path)}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {result_path}")
        
    elif args.video_dir:
        # 批量检测
        print(f"批量检测视频目录: {args.video_dir}")
        batch_results = detector.batch_detect_blur(args.video_dir, args.output_dir)
        
        print(f"批量检测完成:")
        print(f"  总视频数: {batch_results['total_videos']}")
        print(f"  处理成功: {batch_results['processed_videos']}")
        print(f"  检测到模糊: {batch_results['blur_detected_count']}")
        
    else:
        print("请指定 --video_path 或 --video_dir 参数")


if __name__ == "__main__":
    main()
