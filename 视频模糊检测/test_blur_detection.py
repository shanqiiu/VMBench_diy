# -*- coding: utf-8 -*-
"""
视频模糊检测测试脚本
用于测试检测系统的功能
"""

import os
import sys
import json
import tempfile
import numpy as np
import cv2
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_blur_detector import SimpleBlurDetector
from blur_visualization import BlurVisualization
from config import BlurDetectionConfig


def create_test_video(output_path: str, blur_frames: list = None, duration: int = 5, fps: int = 30):
    """
    创建测试视频
    
    Args:
        output_path: 输出视频路径
        blur_frames: 需要添加模糊的帧索引
        duration: 视频时长（秒）
        fps: 帧率
    """
    if blur_frames is None:
        blur_frames = []
    
    # 视频参数
    width, height = 640, 480
    total_frames = duration * fps
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # 创建基础帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加一些内容
        cv2.rectangle(frame, (50, 50), (width-50, height-50), (0, 255, 0), 2)
        cv2.putText(frame, f'Frame {frame_idx}', (100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 添加运动元素
        center_x = width // 2 + int(50 * np.sin(frame_idx * 0.1))
        center_y = height // 2 + int(30 * np.cos(frame_idx * 0.1))
        cv2.circle(frame, (center_x, center_y), 20, (0, 0, 255), -1)
        
        # 添加模糊效果
        if frame_idx in blur_frames:
            # 应用高斯模糊
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        
        out.write(frame)
    
    out.release()
    print(f"测试视频已创建: {output_path}")


def test_simple_detector():
    """测试简化版检测器"""
    print("=== 测试简化版检测器 ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建测试视频
        normal_video = temp_path / "normal_video.mp4"
        blur_video = temp_path / "blur_video.mp4"
        
        # 正常视频（无模糊）
        create_test_video(str(normal_video))
        
        # 模糊视频（第10-15帧模糊）
        create_test_video(str(blur_video), blur_frames=list(range(10, 16)))
        
        try:
            # 初始化检测器（使用CPU避免CUDA问题）
            detector = SimpleBlurDetector(device="cpu")
            
            # 测试正常视频
            print("检测正常视频...")
            normal_result = detector.detect_blur(str(normal_video))
            print(f"正常视频检测结果: {normal_result['blur_severity']} (置信度: {normal_result['confidence']:.3f})")
            
            # 测试模糊视频
            print("检测模糊视频...")
            blur_result = detector.detect_blur(str(blur_video))
            print(f"模糊视频检测结果: {blur_result['blur_severity']} (置信度: {blur_result['confidence']:.3f})")
            
            # 验证结果
            assert normal_result['blur_detected'] == False, "正常视频应该检测为无模糊"
            assert blur_result['blur_detected'] == True, "模糊视频应该检测为有模糊"
            
            print("? 简化版检测器测试通过")
            
        except Exception as e:
            print(f"? 简化版检测器测试失败: {e}")
            return False
    
    return True


def test_visualization():
    """测试可视化功能"""
    print("=== 测试可视化功能 ===")
    
    try:
        # 创建测试数据
        test_result = {
            'video_path': 'test_video.mp4',
            'video_name': 'test_video.mp4',
            'blur_detected': True,
            'confidence': 0.75,
            'blur_severity': '中等模糊',
            'blur_ratio': 0.15,
            'blur_frame_count': 15,
            'total_frames': 100,
            'avg_quality': 0.75,
            'max_quality_drop': 0.12,
            'threshold': 0.025,
            'blur_frames': [10, 15, 20, 25, 30],
            'recommendations': ['建议使用稳定器', '提高录制帧率']
        }
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = BlurVisualization(temp_dir)
            
            # 测试检测报告生成
            report_path = visualizer.create_detection_report(test_result)
            assert os.path.exists(report_path), "检测报告应该被创建"
            
            # 测试批量结果可视化
            batch_results = [test_result, test_result.copy()]
            batch_viz_path = visualizer.visualize_batch_results(batch_results)
            assert os.path.exists(batch_viz_path), "批量可视化应该被创建"
            
            print("? 可视化功能测试通过")
            
    except Exception as e:
        print(f"? 可视化功能测试失败: {e}")
        return False
    
    return True


def test_config():
    """测试配置功能"""
    print("=== 测试配置功能 ===")
    
    try:
        # 测试默认配置
        config = BlurDetectionConfig()
        assert config.get_detection_param('window_size') == 3, "默认窗口大小应该是3"
        assert config.get_detection_param('confidence_threshold') == 0.7, "默认置信度阈值应该是0.7"
        
        # 测试配置更新
        config.update_detection_param('window_size', 5)
        assert config.get_detection_param('window_size') == 5, "配置更新应该生效"
        
        # 测试预设配置
        from config import get_preset_config
        fast_config = get_preset_config('fast')
        assert fast_config.get_detection_param('window_size') == 2, "快速配置窗口大小应该是2"
        
        print("? 配置功能测试通过")
        
    except Exception as e:
        print(f"? 配置功能测试失败: {e}")
        return False
    
    return True


def test_batch_detection():
    """测试批量检测"""
    print("=== 测试批量检测 ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        videos_dir = temp_path / "videos"
        videos_dir.mkdir()
        
        # 创建多个测试视频
        for i in range(3):
            video_path = videos_dir / f"test_video_{i}.mp4"
            if i == 0:
                # 第一个视频无模糊
                create_test_video(str(video_path))
            else:
                # 其他视频有模糊
                blur_frames = list(range(5, 10)) if i == 1 else list(range(15, 25))
                create_test_video(str(video_path), blur_frames=blur_frames)
        
        try:
            # 初始化检测器
            detector = SimpleBlurDetector(device="cpu")
            
            # 批量检测
            results = detector.batch_detect(str(videos_dir), str(temp_path / "results"))
            
            # 验证结果
            assert results['total_videos'] == 3, "应该检测3个视频"
            assert results['processed_videos'] == 3, "应该处理3个视频"
            assert results['blur_detected_count'] >= 2, "应该检测到至少2个模糊视频"
            
            print("? 批量检测测试通过")
            
        except Exception as e:
            print(f"? 批量检测测试失败: {e}")
            return False
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("开始运行视频模糊检测系统测试...")
    
    tests = [
        ("简化版检测器", test_simple_detector),
        ("可视化功能", test_visualization),
        ("配置功能", test_config),
        ("批量检测", test_batch_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"? {test_name} 测试通过")
            else:
                print(f"? {test_name} 测试失败")
        except Exception as e:
            print(f"? {test_name} 测试异常: {e}")
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("? 所有测试通过！")
    else:
        print("?? 部分测试失败，请检查系统配置")


if __name__ == "__main__":
    run_all_tests()
