# CAS注意力权重可视化使用指南

## 概述

本指南介绍如何使用增强版CAS评分系统进行注意力权重可视化，帮助理解视频中哪些区域存在常识违反。

## 功能特性

- ? **注意力权重提取**: 从VideoMAEv2模型提取注意力权重
- ? **热力图可视化**: 生成注意力热力图显示异常区域
- ? **视频帧标注**: 在视频帧上叠加注意力信息
- ? **异常检测**: 自动识别低CAS评分的视频
- ? **批量处理**: 支持批量视频评估和可视化
- ? **详细报告**: 生成完整的分析报告

## 安装依赖

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib seaborn
pip install numpy pandas
```

## 快速开始

### 1. 单视频评估

```python
from enhanced_commonsense_adherence_score import EnhancedCASScorer
import torch

# 初始化参数
args = {
    'model': 'vit_base_patch16_224',
    'input_size': 224,
    'num_frames': 16,
    'tubelet_size': 2,
    'device': 'cuda',
    'enable_visualization': True,
    'visualization_threshold': 0.5
}

# 加载模型
model = create_model(args['model'], ...)
model.to(args['device'])

# 创建CAS评分器
cas_scorer = EnhancedCASScorer(args, model, args['device'])

# 评估单个视频
result = cas_scorer.evaluate_with_visualization('path/to/video.mp4')
print(f"CAS评分: {result['cas_score']:.3f}")
```

### 2. 批量评估

```python
# 批量评估视频目录
results = cas_scorer.batch_evaluate_with_visualization(
    'path/to/video/directory',
    output_dir='./visualization_results'
)
```

### 3. 命令行使用

```bash
# 单视频评估
python enhanced_commonsense_adherence_score.py \
    --video_path path/to/video.mp4 \
    --enable_visualization \
    --output_dir ./results

# 批量评估
python enhanced_commonsense_adherence_score.py \
    --video_dir path/to/videos \
    --enable_visualization \
    --output_dir ./batch_results
```

## 输出结果

### 1. 可视化文件

- `attention_heatmap.png`: 注意力热力图
- `frame_XXX_attention.jpg`: 带注意力标注的视频帧
- `attention_report.txt`: 详细分析报告

### 2. 批量结果

- `cas_visualization_results.json`: 完整评估结果
- `cas_summary.csv`: 评分摘要
- `statistics_report.txt`: 统计报告

## 参数说明

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_visualization` | bool | False | 启用注意力可视化 |
| `--visualization_threshold` | float | 0.5 | 可视化触发阈值 |
| `--output_dir` | str | ./cas_visualization_results | 输出目录 |
| `--video_path` | str | None | 单视频路径 |
| `--video_dir` | str | None | 视频目录路径 |

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | vit_base_patch16_224 | 模型名称 |
| `--input_size` | int | 224 | 输入图像尺寸 |
| `--num_frames` | int | 16 | 视频帧数 |
| `--tubelet_size` | int | 2 | 时间tubelet大小 |

## 使用示例

### 示例1：检测房檐上凝固的雨滴

```python
# 评估包含异常物理现象的视频
result = cas_scorer.evaluate_with_visualization('eaves_frozen_raindrops.mp4')

if result['cas_score'] < 0.5:
    print("检测到低CAS评分，可能存在物理异常")
    print("请查看注意力热力图了解异常区域")
    
    # 查看可视化结果
    if result['visualization']:
        print(f"可视化结果保存在: {result['visualization']['output_dir']}")
```

### 示例2：批量检测异常视频

```python
# 批量评估视频目录
results = cas_scorer.batch_evaluate_with_visualization('./test_videos')

# 分析结果
low_score_videos = [r for r in results if r['cas_score'] < 0.5]
print(f"检测到 {len(low_score_videos)} 个异常视频")

for video in low_score_videos:
    print(f"异常视频: {video['video_path']}, 评分: {video['cas_score']:.3f}")
```

## 可视化结果解读

### 1. 注意力热力图

- **红色区域**: 高注意力，可能是异常区域
- **蓝色区域**: 低注意力，正常区域
- **颜色强度**: 表示注意力权重大小

### 2. 异常检测

- **CAS评分 < 0.3**: 严重违反常识
- **CAS评分 0.3-0.5**: 部分违反常识
- **CAS评分 > 0.5**: 基本符合常识

### 3. 分析报告

报告包含以下信息：
- CAS评分统计
- 注意力分布分析
- 高注意力区域识别
- 异常原因解释

## 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 确保模型路径正确
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **视频加载失败**
   ```bash
   # 检查视频格式和路径
   ffmpeg -i video.mp4 -f null -
   ```

3. **内存不足**
   ```python
   # 减少批处理大小
   args.batch_size = 1
   ```

### 性能优化

1. **GPU加速**: 确保使用CUDA设备
2. **批处理**: 合理设置批处理大小
3. **内存管理**: 及时释放不需要的tensor

## 高级功能

### 1. 自定义注意力分析

```python
# 提取特定层的注意力权重
attention_weights, layer_names, outputs = visualizer.extract_attention_weights(video_tensor)

# 分析特定层的注意力
layer_attention = attention_weights[-1]  # 最后一层
```

### 2. 注意力权重分析

```python
# 计算注意力统计
attention_stats = {
    'mean': np.mean(attention_heatmap),
    'std': np.std(attention_heatmap),
    'max': np.max(attention_heatmap),
    'min': np.min(attention_heatmap)
}
```

## 技术原理

### 1. 注意力机制

VideoMAEv2使用多头自注意力机制：
- 每个patch对应视频中的一个时空区域
- 注意力权重表示不同区域之间的相关性
- 异常区域通常具有异常的注意力模式

### 2. 可视化方法

- **Rollout方法**: 聚合多层注意力权重
- **热力图生成**: 将注意力权重映射到空间位置
- **颜色编码**: 使用颜色表示注意力强度

### 3. 异常检测

- **阈值检测**: 基于CAS评分阈值
- **注意力分析**: 分析注意力分布模式
- **区域标注**: 标注高注意力区域

## 扩展功能

### 1. 自定义可视化

```python
# 自定义颜色映射
def custom_colormap(attention_map):
    # 实现自定义颜色映射逻辑
    pass
```

### 2. 多模态分析

```python
# 结合其他VMBench指标
def multi_modal_analysis(video_path):
    cas_result = cas_scorer.evaluate_with_visualization(video_path)
    # 结合其他指标进行分析
    return combined_analysis
```

## 总结

CAS注意力权重可视化系统提供了强大的视频异常检测和可视化功能，能够：

1. **自动检测**: 识别违反常识的视频内容
2. **精确定位**: 指出具体的异常区域
3. **详细分析**: 提供深入的分析报告
4. **批量处理**: 支持大规模视频评估

通过使用本系统，您可以更好地理解视频生成模型的行为，识别和解决常识违反问题。
