# -*- coding: utf-8 -*-
"""
视频模糊检测使用示例
展示如何使用模糊检测系统
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_blur_detector import SimpleBlurDetector
from blur_visualization import BlurVisualization
from config import BlurDetectionConfig, get_preset_config


def example_single_video_detection():
    """单视频检测示例"""
    print("=== 单视频检测示例 ===")
    
    # 假设有一个视频文件
    video_path = "example_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        print("请将您的视频文件重命名为 'example_video.mp4' 并放在当前目录")
        return
    
    try:
        # 1. 初始化检测器
        print("初始化检测器...")
        detector = SimpleBlurDetector(device="cuda:0")  # 如果有GPU
        
        # 2. 执行检测
        print("开始检测...")
        result = detector.detect_blur(video_path)
        
        # 3. 显示结果
        print(f"\n检测结果:")
        print(f"  视频: {result['video_name']}")
        print(f"  检测到模糊: {result['blur_detected']}")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  模糊严重程度: {result['blur_severity']}")
        print(f"  模糊比例: {result['blur_ratio']:.3f}")
        print(f"  模糊帧数: {result['blur_frame_count']}/{result['total_frames']}")
        
        # 4. 显示建议
        print(f"\n改进建议:")
        for rec in result['recommendations']:
            print(f"  ? {rec}")
        
        # 5. 生成可视化
        print("\n生成可视化结果...")
        visualizer = BlurVisualization("./visualization_results")
        
        # 生成检测报告
        report_path = visualizer.create_detection_report(result)
        print(f"检测报告已保存到: {report_path}")
        
    except Exception as e:
        print(f"检测过程中出现错误: {e}")
        print("请检查模型文件是否正确安装")


def example_batch_detection():
    """批量检测示例"""
    print("\n=== 批量检测示例 ===")
    
    # 假设有一个视频目录
    video_dir = "example_videos"
    
    if not os.path.exists(video_dir):
        print(f"视频目录不存在: {video_dir}")
        print("请创建 'example_videos' 目录并放入一些视频文件")
        return
    
    try:
        # 1. 初始化检测器
        print("初始化检测器...")
        detector = SimpleBlurDetector(device="cuda:0")
        
        # 2. 执行批量检测
        print("开始批量检测...")
        results = detector.batch_detect(video_dir, "./batch_results")
        
        # 3. 显示统计结果
        print(f"\n批量检测结果:")
        print(f"  总视频数: {results['total_videos']}")
        print(f"  处理成功: {results['processed_videos']}")
        print(f"  检测到模糊: {results['blur_detected_count']}")
        
        # 4. 显示每个视频的结果
        print(f"\n各视频检测结果:")
        for result in results['results']:
            if 'error' not in result:
                print(f"  {os.path.basename(result['video_path'])}: {result['blur_severity']} (置信度: {result['confidence']:.3f})")
            else:
                print(f"  {os.path.basename(result['video_path'])}: 处理失败 - {result['error']}")
        
        # 5. 生成批量可视化
        print("\n生成批量可视化...")
        visualizer = BlurVisualization("./batch_visualization")
        batch_viz_path = visualizer.visualize_batch_results(results['results'])
        print(f"批量可视化已保存到: {batch_viz_path}")
        
    except Exception as e:
        print(f"批量检测过程中出现错误: {e}")


def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    try:
        # 1. 创建自定义配置
        config = BlurDetectionConfig()
        
        # 2. 修改检测参数
        config.update_detection_param('window_size', 5)  # 增大窗口
        config.update_detection_param('confidence_threshold', 0.8)  # 提高阈值
        
        # 3. 修改设备配置
        config.update_device_config('device', 'cuda:0')
        
        print("自定义配置:")
        print(f"  窗口大小: {config.get_detection_param('window_size')}")
        print(f"  置信度阈值: {config.get_detection_param('confidence_threshold')}")
        print(f"  计算设备: {config.get_device_config('device')}")
        
        # 4. 使用预设配置
        fast_config = get_preset_config('fast')
        print(f"\n快速配置:")
        print(f"  窗口大小: {fast_config.get_detection_param('window_size')}")
        print(f"  置信度阈值: {fast_config.get_detection_param('confidence_threshold')}")
        
    except Exception as e:
        print(f"配置示例过程中出现错误: {e}")


def example_visualization():
    """可视化示例"""
    print("\n=== 可视化示例 ===")
    
    try:
        # 创建示例数据
        example_result = {
            'video_path': 'example_video.mp4',
            'video_name': 'example_video.mp4',
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
            'quality_scores': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3] * 10,
            'recommendations': ['建议使用稳定器', '提高录制帧率', '确保充足光线']
        }
        
        # 初始化可视化工具
        visualizer = BlurVisualization("./example_visualization")
        
        # 1. 生成检测报告
        print("生成检测报告...")
        report_path = visualizer.create_detection_report(example_result)
        print(f"检测报告: {report_path}")
        
        # 2. 生成质量分数可视化
        print("生成质量分数可视化...")
        quality_viz_path = visualizer.visualize_quality_scores(
            example_result['video_path'],
            example_result['quality_scores'],
            example_result['blur_frames'],
            example_result['threshold']
        )
        print(f"质量分数可视化: {quality_viz_path}")
        
        # 3. 生成批量结果可视化
        print("生成批量结果可视化...")
        batch_results = [example_result, example_result.copy()]
        batch_viz_path = visualizer.visualize_batch_results(batch_results)
        print(f"批量结果可视化: {batch_viz_path}")
        
    except Exception as e:
        print(f"可视化示例过程中出现错误: {e}")


def main():
    """主函数"""
    print("视频模糊检测系统使用示例")
    print("=" * 50)
    
    # 运行各种示例
    example_single_video_detection()
    example_batch_detection()
    example_custom_config()
    example_visualization()
    
    print("\n" + "=" * 50)
    print("示例运行完成！")
    print("\n要运行实际的检测，请:")
    print("1. 准备您的视频文件")
    print("2. 运行: python run_blur_detection.py --video_path your_video.mp4")
    print("3. 或运行: python run_blur_detection.py --video_dir your_video_directory")


if __name__ == "__main__":
    main()
