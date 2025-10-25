# Q-Align算法实现逻辑详解

## 1. 算法概述

Q-Align是一个基于多模态大语言模型的视频质量评估算法，主要用于评估视频的感知质量。它结合了视觉编码器、语言模型和偏好学习机制。

## 2. 核心架构

### 2.1 模型组成

```
Q-Align = 视觉编码器 + 视觉抽象器 + 语言模型 + 偏好学习头
```

- **视觉编码器**: 基于CLIP的视觉特征提取
- **视觉抽象器**: 将视觉特征转换为语言模型可理解的表示
- **语言模型**: 基于LLaMA的文本理解
- **偏好学习头**: 将质量评估转换为偏好选择

### 2.2 技术栈

- **基础模型**: mPLUG-Owl2 (多模态大语言模型)
- **视觉编码**: CLIP ViT
- **语言模型**: LLaMA-2
- **训练框架**: Transformers + LoRA微调

## 3. 输入要求

### 3.1 输入格式

```python
# 输入数据结构
input_data = {
    'video_frames': List[List[PIL.Image]],  # 视频帧序列
    'prompt': str,                         # 评估提示
    'window_size': int                     # 滑动窗口大小
}
```

### 3.2 具体输入要求

#### 3.2.1 视频帧输入
```python
# 视频帧要求
video_frames = [
    [frame1, frame2, frame3, ...],  # 第一个时间窗口
    [frame2, frame3, frame4, ...],  # 第二个时间窗口
    ...
]

# 帧格式要求
- 类型: PIL.Image
- 颜色空间: RGB
- 尺寸: 任意（会自动调整）
- 数量: 根据window_size确定
```

#### 3.2.2 评估提示
```python
# 默认提示
prompt = "USER: How would you rate the quality of this video?\n<|image|>\nASSISTANT: The quality of the video is"

# 提示结构
- USER: 用户问题
- <|image|>: 图像占位符
- ASSISTANT: 模型回答
```

#### 3.2.3 滑动窗口参数
```python
window_size = 3  # 每个窗口包含3帧
# 窗口滑动方式：
# 帧1: [frame1, frame1, frame1]  # 开始帧重复
# 帧2: [frame1, frame2, frame3]  # 正常窗口
# 帧3: [frame2, frame3, frame4]  # 滑动窗口
# ...
```

## 4. 算法实现流程

### 4.1 初始化阶段

```python
class QAlignVideoScorer(nn.Module):
    def __init__(self, pretrained="q-future/one-align", device="cuda:0"):
        # 1. 加载预训练模型
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained, None, "mplug_owl2", device=device
        )
        
        # 2. 设置评估提示
        prompt = "USER: How would you rate the quality of this video?\n<|image|>\nASSISTANT: The quality of the video is"
        
        # 3. 定义质量等级和权重
        self.preferential_ids_ = [id_[1] for id_ in tokenizer([
            "excellent", "good", "fair", "poor", "bad"
        ])["input_ids"]]
        
        self.weight_tensor = torch.Tensor([1, 0.75, 0.5, 0.25, 0.]).half()
        
        # 4. 预处理输入
        self.input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)
```

### 4.2 视频预处理

```python
def load_video_sliding_window(video_file, window_size=5):
    """
    滑动窗口方式读取视频
    
    Args:
        video_file: 视频文件路径
        window_size: 窗口大小
    
    Returns:
        frames_by_group: 按窗口组织的帧序列
    """
    vr = VideoReader(video_file)
    total_frames = len(vr)
    frames_by_group = []
    
    # 计算窗口扩展
    left_extend = (window_size - 1) // 2
    right_extend = window_size - 1 - left_extend
    
    for current_frame in range(total_frames):
        # 计算窗口范围
        start_frame = max(0, current_frame - left_extend)
        end_frame = min(total_frames, current_frame + right_extend + 1)
        
        frame_indices = list(range(start_frame, end_frame))
        
        # 填充不足的帧
        while len(frame_indices) < window_size:
            if start_frame == 0:
                frame_indices.append(frame_indices[-1])  # 重复最后一帧
            else:
                frame_indices.insert(0, frame_indices[0])  # 重复第一帧
        
        # 获取帧数据
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # 转换为PIL图像
        if current_frame < left_extend:
            frames_by_group.append([Image.fromarray(frames[0])] * window_size)
        else:
            frames_by_group.append([Image.fromarray(frame) for frame in frames])
    
    return frames_by_group
```

### 4.3 图像预处理

```python
def expand2square(self, pil_img, background_color):
    """
    将图像扩展为正方形
    
    Args:
        pil_img: PIL图像
        background_color: 背景颜色
    
    Returns:
        正方形PIL图像
    """
    width, height = pil_img.size
    
    if width == height:
        return pil_img
    elif width > height:
        # 高度小于宽度，在上下添加背景
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        # 宽度小于高度，在左右添加背景
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
```

### 4.4 前向推理

```python
def forward(self, video: List[List[Image.Image]]):
    """
    前向推理过程
    
    Args:
        video: 视频帧序列 [batch_size, window_size, PIL.Image]
    
    Returns:
        output_logits: 原始logits
        softmax_probs: 软最大概率
        weighted_scores: 加权分数
    """
    # 1. 图像预处理
    video = [[self.expand2square(
        frame, 
        tuple(int(x*255) for x in self.image_processor.image_mean)
    ) for frame in vid] for vid in video]
    
    with torch.inference_mode():
        # 2. 图像编码
        video_tensors = [
            self.image_processor.preprocess(vid, return_tensors="pt")["pixel_values"]
            .half().to(self.model.device) 
            for vid in video
        ]
        
        # 3. 文本编码
        input_tensors = self.input_ids.repeat(len(video_tensors), 1)
        
        # 4. 多模态推理
        output = self.model(input_tensors, images=video_tensors)
        
        # 5. 提取质量评估logits
        output_logits = output["logits"][:, -1, self.preferential_ids_]
        
        # 6. 计算概率分布
        softmax_probs = torch.softmax(output_logits, -1)
        
        # 7. 计算加权分数
        weighted_scores = softmax_probs @ self.weight_tensor
        
        return output_logits, softmax_probs, weighted_scores
```

## 5. 质量评估机制

### 5.1 偏好学习

```python
# 质量等级定义
quality_levels = ["excellent", "good", "fair", "poor", "bad"]

# 对应权重
weights = [1.0, 0.75, 0.5, 0.25, 0.0]

# 最终分数计算
final_score = Σ(probability_i × weight_i)
```

### 5.2 分数解释

- **1.0**: 优秀质量
- **0.75**: 良好质量  
- **0.5**: 一般质量
- **0.25**: 较差质量
- **0.0**: 很差质量

## 6. 模糊检测应用

### 6.1 质量分数分析

```python
def get_artifacts_frames(scores, threshold=0.025):
    """
    基于质量分数差异检测异常帧
    
    Args:
        scores: 质量分数序列
        threshold: 检测阈值
    
    Returns:
        artifact_indices: 异常帧索引
    """
    # 计算相邻帧分数差异
    score_diffs = np.abs(np.diff(scores))
    
    # 识别超过阈值的帧
    artifact_indices = np.where(score_diffs > threshold)[0]
    
    # 返回异常帧（包括当前帧和下一帧）
    artifacts_frames = np.unique(np.concatenate([
        artifact_indices, 
        artifact_indices + 1
    ]))
    
    return artifacts_frames
```

### 6.2 自适应阈值

```python
def set_threshold(camera_movement):
    """
    根据相机运动幅度设置自适应阈值
    
    Args:
        camera_movement: 相机运动幅度 (0-1)
    
    Returns:
        threshold: 检测阈值
    """
    if camera_movement is None:
        return 0.01
    elif camera_movement < 0.1:
        return 0.01      # 静态场景，低阈值
    elif camera_movement < 0.3:
        return 0.015     # 轻微运动，中等阈值
    elif camera_movement < 0.5:
        return 0.025     # 中等运动，较高阈值
    else:
        return 0.03      # 剧烈运动，高阈值
```

## 7. 性能优化

### 7.1 内存优化

```python
# 使用半精度推理
model.half()

# 推理模式
with torch.inference_mode():
    output = model(inputs)
```

### 7.2 批处理优化

```python
# 批量处理多个视频窗口
batch_size = 8
for i in range(0, len(video_windows), batch_size):
    batch = video_windows[i:i+batch_size]
    scores = model(batch)
```

## 8. 使用示例

### 8.1 基本使用

```python
from motion_smoothness_score import QAlignVideoScorer, load_video_sliding_window

# 初始化模型
scorer = QAlignVideoScorer(pretrained="q-future/one-align", device="cuda:0")

# 加载视频
video_frames = load_video_sliding_window("video.mp4", window_size=3)

# 评估质量
_, _, scores = scorer(video_frames)
print(f"质量分数: {scores}")
```

### 8.2 模糊检测

```python
# 检测模糊帧
blur_frames = get_artifacts_frames(scores, threshold=0.025)
print(f"检测到模糊帧: {blur_frames}")

# 计算MSS分数
mss_score = 1 - len(blur_frames) / len(scores)
print(f"MSS分数: {mss_score}")
```

## 9. 算法优势

1. **感知对齐**: 基于人类感知的质量评估
2. **多模态理解**: 结合视觉和语言信息
3. **自适应检测**: 根据场景动态调整阈值
4. **高效推理**: 优化的模型架构和推理流程
5. **可解释性**: 提供详细的质量分析结果

## 10. 局限性

1. **计算资源**: 需要GPU加速
2. **模型大小**: 预训练模型较大
3. **实时性**: 推理速度相对较慢
4. **泛化性**: 对特定场景的适应性有限

## 11. 改进方向

1. **模型压缩**: 减少模型大小和计算量
2. **实时优化**: 提高推理速度
3. **多尺度检测**: 支持不同分辨率的视频
4. **领域适应**: 针对特定场景的微调
