# VMBench 代码解读文档

## 项目概述

VMBench (Video Motion Benchmark) 是一个用于感知对齐的视频运动生成的综合基准测试工具。该项目专注于评估视频生成模型在运动质量方面的表现，通过五个核心维度提供细粒度的评估指标。

### 核心特性

1. **感知驱动的运动评估指标** - 基于人类感知的五个维度评估视频运动质量
2. **元引导的运动提示生成** - 结构化的方法生成多样化的运动提示
3. **人类对齐的验证机制** - 提供人类偏好注释来验证基准测试

## 项目结构

```
VMBench/
├── 核心评估模块/
│   ├── perceptible_amplitude_score.py      # 可感知幅度评分 (PAS)
│   ├── object_integrity_score.py           # 对象完整性评分 (OIS)
│   ├── temporal_coherence_score.py         # 时间一致性评分 (TCS)
│   ├── motion_smoothness_score.py          # 运动平滑度评分 (MSS)
│   └── commonsense_adherence_score.py      # 常识遵循评分 (CAS)
├── 工具模块/
│   ├── bench_utils/                        # 评估工具集
│   │   ├── create_meta_info.py            # 创建元信息
│   │   ├── calculate_score.py             # 计算平均分数
│   │   ├── pose_utils.py                  # 姿态分析工具
│   │   ├── tcs_utils.py                   # 时间一致性工具
│   │   └── cas_utils.py                   # 常识遵循工具
│   └── sample_video_demo.py               # 视频生成示例
├── 第三方依赖/
│   ├── Grounded-Segment-Anything/         # 对象检测和分割
│   ├── Grounded-SAM-2/                    # 高级分割模型
│   ├── co-tracker/                        # 对象跟踪
│   ├── mmpose/                            # 姿态估计
│   ├── Q-Align/                           # 质量评估
│   └── VideoMAEv2/                        # 视频理解模型
├── 提示数据/
│   └── prompts/                           # 1050个测试提示
└── 评估脚本/
    └── evaluate.sh                        # 完整评估流程
```

## 核心评估模块详解

### 1. 可感知幅度评分 (PAS) - `perceptible_amplitude_score.py`

**功能**: 评估视频中主体对象的运动幅度是否足够明显和可感知。

**核心算法**:
```python
def calculate_motion_degree(keypoints, video_width, video_height):
    # 计算视频对角线长度
    diagonal = torch.sqrt(torch.tensor(video_width**2 + video_height**2))
    
    # 计算相邻帧间的欧几里得距离
    distances = torch.norm(keypoints[:, 1:] - keypoints[:, :-1], dim=3)
    
    # 归一化距离
    normalized_distances = distances / diagonal
    
    # 计算总归一化运动距离
    total_normalized_distances = torch.sum(normalized_distances, dim=1)
    
    # 计算平均运动幅度
    motion_amplitudes = torch.mean(total_normalized_distances, dim=1)
    
    return motion_amplitudes
```

**技术栈**:
- GroundingDINO: 对象检测
- SAM (Segment Anything Model): 对象分割
- Co-Tracker: 运动跟踪

**评估流程**:
1. 使用GroundingDINO检测目标对象
2. 使用SAM生成精确的对象掩码
3. 分离主体和背景区域
4. 使用Co-Tracker跟踪运动轨迹
5. 计算归一化的运动幅度

### 2. 对象完整性评分 (OIS) - `object_integrity_score.py`

**功能**: 评估视频中人体姿态的完整性和一致性，检测不合理的身体部位变化。

**核心算法**:
```python
def analyze_lengths_over_time(instance_info, threshold=0.45):
    # 分析身体部位长度变化
    body_parts = [
        ('torso', 5, 11),           # 躯干
        ('left_upper_arm', 5, 7),   # 左上臂
        ('left_forearm', 7, 9),     # 左前臂
        # ... 更多身体部位
    ]
    
    # 使用滑动窗口分析长度变化
    for part_name in body_parts:
        # 计算相对变化
        change = abs(window_averages[i+1] - window_averages[i]) / window_averages[i]
        if change > threshold:
            # 检测到异常变化
            anomalies.append(f"{part_name} shows significant change")
```

**技术栈**:
- MMPose: 人体姿态估计
- RTMPose: 实时姿态检测
- 统计分析: 检测异常的身体部位变化

**评估维度**:
- 身体部位长度一致性
- 关节角度合理性
- 姿态稳定性

### 3. 时间一致性评分 (TCS) - `temporal_coherence_score.py`

**功能**: 评估视频中对象的出现和消失是否符合物理规律和时间逻辑。

**核心算法**:
```python
def get_disappear_objects(tracking_result):
    # 检测突然消失的对象
    for i in range(len(tracking_result) - 1):
        dict1 = tracking_result[i]
        dict2 = tracking_result[i + 1]
        
        # 找出消失的键
        disappeared_keys = set(dict1.keys()) - set(dict2.keys())
        
        # 分析消失原因
        for key in disappeared_keys:
            # 检查是否从边缘消失
            edge_vanish = is_edge_vanish(pred_tracks, pred_visibility, ...)
            # 检查是否因过小而消失
            small_vanish = is_small_vanish(pred_tracks, pred_visibility, ...)
            # 检查是否为检测错误
            detect_error = is_vanish_detect_error(pred_tracks, pred_visibility, ...)
```

**技术栈**:
- Grounded-SAM-2: 高级对象分割和跟踪
- Co-Tracker: 运动轨迹分析
- 物理规律验证

**评估标准**:
- 对象出现/消失的合理性
- 边缘消失检测
- 尺寸变化分析
- 检测错误识别

### 4. 运动平滑度评分 (MSS) - `motion_smoothness_score.py`

**功能**: 评估视频中运动的质量和流畅性，检测运动中的伪影和不连续性。

**核心算法**:
```python
def get_artifacts_frames(scores, threshold=0.025):
    # 计算相邻帧间的分数差异
    score_diffs = np.abs(np.diff(scores))
    
    # 识别分数差异超过阈值的帧
    artifact_indices = np.where(score_diffs > threshold)[0]
    
    # 返回有问题的帧
    artifacts_frames = np.unique(np.concatenate([artifact_indices, artifact_indices + 1]))
    
    return artifacts_frames

def set_threshold(camera_movement):
    # 根据相机运动幅度设置阈值
    if camera_movement < 0.1:
        return 0.01
    elif 0.1 <= camera_movement < 0.3:
        return 0.015
    # ... 更多阈值设置
```

**技术栈**:
- Q-Align: 视频质量评估模型
- 滑动窗口分析
- 自适应阈值设置

**评估方法**:
- 使用Q-Align模型评估每帧质量
- 基于相机运动调整检测阈值
- 识别质量异常的帧序列

### 5. 常识遵循评分 (CAS) - `commonsense_adherence_score.py`

**功能**: 评估视频内容是否符合物理常识和现实逻辑。

**核心算法**:
```python
def final_merge(eval_path, num_tasks, meta_info_path, method='prob'):
    # 合并多个进程的评估结果
    for row in ans:
        video_id, prob, _, _, _, _ = row
        # 计算加权总分
        prob_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        total_score = np.sum(prob * prob_weights)
        item['commonsense_adherence_score'] = total_score
```

**技术栈**:
- VideoMAEv2: 视频理解模型
- 分布式评估
- 多进程结果合并

**评估维度**:
- 物理规律遵循
- 现实逻辑一致性
- 常识合理性

## 工具模块详解

### 元信息管理 - `create_meta_info.py`

**功能**: 创建和管理视频评估的元信息文件。

```python
def create_new_json(json_data, video_folder):
    new_json_data = []
    for item in json_data:
        index = item['index']
        video_filename = f"{index}.mp4"
        video_path = os.path.join(video_folder, video_filename)
        if os.path.exists(video_path):
            new_item = item.copy()
            new_item['filepath'] = os.path.abspath(video_path)
            new_json_data.append(new_item)
    return new_json_data
```

### 分数计算 - `calculate_score.py`

**功能**: 计算各维度的平均分数并生成CSV报告。

```python
def calculate_averages(json_file, output_csv):
    scores = {
        'perceptible_amplitude_socre': [],
        'object_integrity_score': [],
        'temporal_coherence_score': [],
        'motion_smoothness_score': [],
        'commonsense_adherence_score': []
    }
    
    # 计算平均分数
    averages = {key: mean(values) for key, values in scores.items() if values}
    total_score = mean(averages.values()) if averages else 0
```

### 姿态分析工具 - `pose_utils.py`

**功能**: 分析人体姿态的完整性和一致性。

**核心功能**:
- `analyze_lengths_over_time()`: 分析身体部位长度变化
- `analyze_joint_angles()`: 分析关节角度变化
- 异常检测和评分计算

## 评估流程

### 完整评估流程 - `evaluate.sh`

```bash
#!/bin/bash
VIDEO_DIR=$1
current_time=$(date "+%Y%m%d_%H%M%S")
META_INFO_PATH="./eval_results/${current_time}/results.json"

# 1. 创建元信息
python bench_utils/create_meta_info.py -v $VIDEO_DIR -o $META_INFO_PATH

# 2. PAS 评估
python perceptible_amplitude_score.py --meta_info_path $META_INFO_PATH

# 3. OIS 评估
python object_integrity_score.py --meta-info-path $META_INFO_PATH

# 4. TCS 评估
python temporal_coherence_score.py --meta_info_path $META_INFO_PATH

# 5. CAS 评估 (分布式)
torchrun --nproc_per_node=1 commonsense_adherence_score.py \
    --model vit_giant_patch14_224 \
    --data_set Commonsense-Adherence \
    --meta_info_path ${META_INFO_PATH}

# 6. MSS 评估 (依赖PAS结果)
python motion_smoothness_score.py --meta_info_path $META_INFO_PATH

# 7. 生成最终报告
python bench_utils/calculate_score.py -i $META_INFO_PATH -o $META_INFO_DIR"/scores.csv"
```

## 依赖关系

### 核心依赖
- **PyTorch**: 深度学习框架
- **OpenCV**: 视频处理
- **NumPy**: 数值计算
- **PIL**: 图像处理

### 第三方模型
- **GroundedDINO**: 对象检测
- **SAM/SAM2**: 对象分割
- **Co-Tracker**: 运动跟踪
- **MMPose**: 姿态估计
- **Q-Align**: 质量评估
- **VideoMAEv2**: 视频理解

### 安装要求
```bash
# 基础环境
conda create -n VMBench python=3.10
pip install torch==2.5.1 torchvision==0.20.1

# 各模块安装
cd Grounded-Segment-Anything && pip install -e .
cd ../Grounded-SAM-2 && pip install -e .
cd ../mmpose && pip install -e .
cd ../Q-Align && pip install -e .
cd ../VideoMAEv2 && pip install -r requirements.txt
```

## 使用示例

### 1. 视频生成
```python
# 使用示例模型生成视频
from diffusers import CogVideoXPipeline

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b")
video = pipe(prompt="A green turtle swims alongside dolphins", num_frames=49)
```

### 2. 评估执行
```bash
# 准备视频文件
python sample_video_demo.py --prompt_path ./prompts/prompts.json --save_dir ./eval_results/videos

# 运行完整评估
bash evaluate.sh ./eval_results/videos
```

### 3. 结果分析
```python
# 读取评估结果
import json
with open('./eval_results/20250101_120000/scores.csv', 'r') as f:
    results = json.load(f)
    
# 分析各维度分数
print(f"PAS: {results['perceptible_amplitude_score']}")
print(f"OIS: {results['object_integrity_score']}")
print(f"TCS: {results['temporal_coherence_score']}")
print(f"MSS: {results['motion_smoothness_score']}")
print(f"CAS: {results['commonsense_adherence_score']}")
```

## 技术特点

### 1. 多维度评估
- **PAS**: 运动幅度感知
- **OIS**: 对象完整性
- **TCS**: 时间一致性
- **MSS**: 运动平滑度
- **CAS**: 常识遵循

### 2. 感知对齐
- 基于人类感知的评估标准
- 35.3%的Spearman相关性提升
- 人类偏好验证机制

### 3. 技术先进性
- 集成最新的计算机视觉模型
- 分布式评估支持
- 自适应阈值调整
- 多进程并行处理

### 4. 可扩展性
- 模块化设计
- 易于添加新的评估维度
- 支持自定义评估标准

## 性能指标

### 评估效率
基于CogVideoX-5B模型的1050个视频评估时间：

| 评估维度 | 耗时 | 说明 |
|---------|------|------|
| PAS | 45分钟 | 对象检测和运动跟踪 |
| OIS | 30分钟 | 姿态估计和分析 |
| TCS | 2小时 | 对象跟踪和一致性分析 |
| MSS | 2.5小时 | 质量评估和伪影检测 |
| CAS | 1小时 | 常识遵循评估 |
| **总计** | **6小时45分钟** | **完整评估流程** |

### 模型性能对比
| 模型 | 平均分 | PAS | OIS | TCS | MSS | CAS |
|------|--------|-----|-----|-----|-----|-----|
| OpenSora-v1.2 | 51.6 | 31.2 | 61.9 | 73.0 | 3.4 | 88.5 |
| Mochi 1 | 53.2 | 37.7 | 62.0 | 68.6 | 14.4 | 83.6 |
| OpenSora-Plan-v1.3.0 | 58.9 | 39.3 | 76.0 | 78.6 | 6.0 | 94.7 |
| CogVideoX-5B | 60.6 | 50.6 | 61.6 | 75.4 | 24.6 | 91.0 |
| HunyuanVideo | 63.4 | 51.9 | 81.6 | 65.8 | 26.1 | 96.3 |
| Wan2.1 | **78.4** | **62.8** | **84.2** | 66.0 | 17.9 | **97.8** |

## 总结

VMBench是一个全面的视频运动质量评估基准测试工具，通过五个核心维度提供了细粒度的评估能力。其感知对齐的设计理念和先进的技术栈使其成为视频生成模型评估的重要工具。模块化的架构和丰富的功能使其既适合研究使用，也适合实际应用部署。

该项目的成功在于：
1. **全面的评估覆盖**: 从运动幅度到常识遵循的全方位评估
2. **感知对齐设计**: 基于人类感知的评估标准
3. **技术先进性**: 集成最新的计算机视觉和深度学习技术
4. **实用性强**: 提供完整的评估流程和工具链
5. **可扩展性**: 模块化设计便于扩展和维护
