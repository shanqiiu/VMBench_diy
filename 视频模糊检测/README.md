# 视频模糊检测系统

基于VMBench的视频模糊检测系统，主要使用MSS (运动平滑度评分) 和 PAS (可感知幅度评分) 进行模糊检测。

## 功能特性

- **基于VMBench**: 利用VMBench的MSS和PAS评分器进行模糊检测
- **自适应阈值**: 根据相机运动幅度自动调整检测阈值
- **多级检测**: 支持轻微、中等、严重模糊的区分
- **可视化支持**: 提供丰富的可视化结果
- **批量处理**: 支持批量视频检测
- **配置灵活**: 支持多种预设配置

## 系统架构

```
视频模糊检测/
├── blur_detection_pipeline.py    # 完整版检测管道
├── simple_blur_detector.py       # 简化版检测器
├── blur_visualization.py         # 可视化工具
├── config.py                     # 配置管理
├── run_blur_detection.py         # 运行脚本
└── README.md                     # 说明文档
```

## 安装要求

### 基础依赖
```bash
pip install torch torchvision
pip install opencv-python
pip install numpy matplotlib seaborn
pip install pillow tqdm
```

### VMBench依赖
确保已安装VMBench的完整环境，包括：
- Q-Align模型
- GroundingDINO
- SAM (Segment Anything Model)
- Co-Tracker

## 快速开始

### 1. 单视频检测

```bash
# 使用简化版检测器（推荐）
python run_blur_detection.py --video_path /path/to/video.mp4

# 使用完整版检测器
python run_blur_detection.py --video_path /path/to/video.mp4 --use_simple=False
```

### 2. 批量检测

```bash
# 批量检测视频目录
python run_blur_detection.py --video_dir /path/to/video/directory

# 指定输出目录
python run_blur_detection.py --video_dir /path/to/videos --output_dir ./results
```

### 3. 配置预设

```bash
# 快速检测（速度优先）
python run_blur_detection.py --video_path /path/to/video.mp4 --config_preset fast

# 精确检测（精度优先）
python run_blur_detection.py --video_path /path/to/video.mp4 --config_preset accurate

# 平衡检测（默认）
python run_blur_detection.py --video_path /path/to/video.mp4 --config_preset balanced
```

## 详细使用

### 1. 简化版检测器

```python
from simple_blur_detector import SimpleBlurDetector

# 初始化检测器
detector = SimpleBlurDetector(device="cuda:0")

# 检测单个视频
result = detector.detect_blur("video.mp4")

# 批量检测
results = detector.batch_detect("/path/to/videos", "./results")
```

### 2. 完整版检测器

```python
from blur_detection_pipeline import BlurDetectionPipeline

# 初始化检测器
detector = BlurDetectionPipeline(device="cuda:0")

# 检测单个视频
result = detector.detect_blur_in_video("video.mp4", subject_noun="person")

# 批量检测
results = detector.batch_detect_blur("/path/to/videos", "./results")
```

### 3. 可视化工具

```python
from blur_visualization import BlurVisualization

# 初始化可视化工具
visualizer = BlurVisualization("./visualizations")

# 可视化质量分数
visualizer.visualize_quality_scores(video_path, quality_scores, blur_frames, threshold)

# 可视化模糊帧
visualizer.visualize_blur_frames(video_path, blur_frames)

# 生成检测报告
visualizer.create_detection_report(result)
```

## 配置说明

### 检测参数

```python
from config import BlurDetectionConfig

config = BlurDetectionConfig()

# 更新检测参数
config.update_detection_param('window_size', 5)
config.update_detection_param('confidence_threshold', 0.8)

# 更新模型路径
config.update_model_path('q_align_model', '/path/to/q-align')
```

### 预设配置

- **fast**: 快速检测，适合实时应用
- **accurate**: 精确检测，适合质量要求高的场景
- **balanced**: 平衡检测，默认推荐配置

## 输出结果

### 检测结果格式

```json
{
  "video_path": "/path/to/video.mp4",
  "video_name": "video.mp4",
  "blur_detected": true,
  "confidence": 0.85,
  "blur_severity": "中等模糊",
  "blur_ratio": 0.15,
  "blur_frame_count": 15,
  "total_frames": 100,
  "avg_quality": 0.75,
  "max_quality_drop": 0.12,
  "threshold": 0.025,
  "blur_frames": [10, 15, 20, 25, 30],
  "recommendations": [
    "建议使用稳定器",
    "提高录制帧率",
    "确保充足光线"
  ]
}
```

### 可视化输出

- **质量分数变化图**: 显示视频质量分数随时间的变化
- **模糊帧展示**: 展示检测到的模糊帧
- **检测报告**: 综合的检测结果报告
- **批量统计**: 批量检测的统计结果

## 算法原理

### 1. MSS评分器 (主要检测)

基于Q-Align模型评估视频质量：
1. 使用滑动窗口提取视频帧
2. 通过Q-Align模型计算每帧质量分数
3. 计算相邻帧间质量分数差异
4. 根据相机运动幅度设置自适应阈值
5. 识别质量异常下降的帧作为模糊帧

### 2. PAS评分器 (辅助验证)

基于运动跟踪验证模糊：
1. 使用GroundingDINO检测主体对象
2. 使用SAM生成精确对象掩码
3. 使用Co-Tracker跟踪运动轨迹
4. 计算归一化运动幅度
5. 模糊会导致运动跟踪不准确，运动幅度异常低

### 3. 综合判断

```python
# 综合置信度计算
confidence = mss_score * 0.8 + pas_score * 0.2

# 模糊检测判断
blur_detected = (
    len(blur_frames) > 0 and 
    confidence < confidence_threshold
)
```

## 性能优化

### 1. 模型优化
- 使用简化版检测器减少计算量
- 调整滑动窗口大小平衡速度和精度
- 使用GPU加速计算

### 2. 批处理优化
- 并行处理多个视频
- 内存管理优化
- 结果缓存机制

### 3. 配置调优
- 根据场景选择合适的配置预设
- 调整检测阈值平衡误报和漏报
- 优化可视化参数

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   解决方案: 检查模型文件路径，确保已下载所有预训练模型
   ```

2. **CUDA内存不足**
   ```
   解决方案: 减少batch_size，或使用CPU模式
   ```

3. **检测精度不高**
   ```
   解决方案: 使用accurate配置，或调整检测阈值
   ```

4. **处理速度慢**
   ```
   解决方案: 使用fast配置，或减少视频分辨率
   ```

### 调试模式

```bash
# 启用详细日志
python run_blur_detection.py --video_path video.mp4 --verbose

# 跳过可视化加速处理
python run_blur_detection.py --video_path video.mp4 --no_visualization
```

## 扩展开发

### 添加新的检测算法

```python
class CustomBlurDetector(SimpleBlurDetector):
    def detect_blur(self, video_path):
        # 实现自定义检测逻辑
        pass
```

### 自定义可视化

```python
class CustomVisualization(BlurVisualization):
    def custom_visualization(self, results):
        # 实现自定义可视化
        pass
```

## 许可证

本项目基于VMBench开发，遵循相同的许可证条款。

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 更新日志

- v1.0.0: 初始版本，支持基础模糊检测
- v1.1.0: 添加可视化功能
- v1.2.0: 优化性能，添加批量处理
