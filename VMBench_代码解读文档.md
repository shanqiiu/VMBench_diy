# VMBench �������ĵ�

## ��Ŀ����

VMBench (Video Motion Benchmark) ��һ�����ڸ�֪�������Ƶ�˶����ɵ��ۺϻ�׼���Թ��ߡ�����Ŀרע��������Ƶ����ģ�����˶���������ı��֣�ͨ���������ά���ṩϸ���ȵ�����ָ�ꡣ

### ��������

1. **��֪�������˶�����ָ��** - ���������֪�����ά��������Ƶ�˶�����
2. **Ԫ�������˶���ʾ����** - �ṹ���ķ������ɶ��������˶���ʾ
3. **����������֤����** - �ṩ����ƫ��ע������֤��׼����

## ��Ŀ�ṹ

```
VMBench/
������ ��������ģ��/
��   ������ perceptible_amplitude_score.py      # �ɸ�֪�������� (PAS)
��   ������ object_integrity_score.py           # �������������� (OIS)
��   ������ temporal_coherence_score.py         # ʱ��һ�������� (TCS)
��   ������ motion_smoothness_score.py          # �˶�ƽ�������� (MSS)
��   ������ commonsense_adherence_score.py      # ��ʶ��ѭ���� (CAS)
������ ����ģ��/
��   ������ bench_utils/                        # �������߼�
��   ��   ������ create_meta_info.py            # ����Ԫ��Ϣ
��   ��   ������ calculate_score.py             # ����ƽ������
��   ��   ������ pose_utils.py                  # ��̬��������
��   ��   ������ tcs_utils.py                   # ʱ��һ���Թ���
��   ��   ������ cas_utils.py                   # ��ʶ��ѭ����
��   ������ sample_video_demo.py               # ��Ƶ����ʾ��
������ ����������/
��   ������ Grounded-Segment-Anything/         # ������ͷָ�
��   ������ Grounded-SAM-2/                    # �߼��ָ�ģ��
��   ������ co-tracker/                        # �������
��   ������ mmpose/                            # ��̬����
��   ������ Q-Align/                           # ��������
��   ������ VideoMAEv2/                        # ��Ƶ���ģ��
������ ��ʾ����/
��   ������ prompts/                           # 1050��������ʾ
������ �����ű�/
    ������ evaluate.sh                        # ������������
```

## ��������ģ�����

### 1. �ɸ�֪�������� (PAS) - `perceptible_amplitude_score.py`

**����**: ������Ƶ�����������˶������Ƿ��㹻���ԺͿɸ�֪��

**�����㷨**:
```python
def calculate_motion_degree(keypoints, video_width, video_height):
    # ������Ƶ�Խ��߳���
    diagonal = torch.sqrt(torch.tensor(video_width**2 + video_height**2))
    
    # ��������֡���ŷ����þ���
    distances = torch.norm(keypoints[:, 1:] - keypoints[:, :-1], dim=3)
    
    # ��һ������
    normalized_distances = distances / diagonal
    
    # �����ܹ�һ���˶�����
    total_normalized_distances = torch.sum(normalized_distances, dim=1)
    
    # ����ƽ���˶�����
    motion_amplitudes = torch.mean(total_normalized_distances, dim=1)
    
    return motion_amplitudes
```

**����ջ**:
- GroundingDINO: ������
- SAM (Segment Anything Model): ����ָ�
- Co-Tracker: �˶�����

**��������**:
1. ʹ��GroundingDINO���Ŀ�����
2. ʹ��SAM���ɾ�ȷ�Ķ�������
3. ��������ͱ�������
4. ʹ��Co-Tracker�����˶��켣
5. �����һ�����˶�����

### 2. �������������� (OIS) - `object_integrity_score.py`

**����**: ������Ƶ��������̬�������Ժ�һ���ԣ���ⲻ��������岿λ�仯��

**�����㷨**:
```python
def analyze_lengths_over_time(instance_info, threshold=0.45):
    # �������岿λ���ȱ仯
    body_parts = [
        ('torso', 5, 11),           # ����
        ('left_upper_arm', 5, 7),   # ���ϱ�
        ('left_forearm', 7, 9),     # ��ǰ��
        # ... �������岿λ
    ]
    
    # ʹ�û������ڷ������ȱ仯
    for part_name in body_parts:
        # ������Ա仯
        change = abs(window_averages[i+1] - window_averages[i]) / window_averages[i]
        if change > threshold:
            # ��⵽�쳣�仯
            anomalies.append(f"{part_name} shows significant change")
```

**����ջ**:
- MMPose: ������̬����
- RTMPose: ʵʱ��̬���
- ͳ�Ʒ���: ����쳣�����岿λ�仯

**����ά��**:
- ���岿λ����һ����
- �ؽڽǶȺ�����
- ��̬�ȶ���

### 3. ʱ��һ�������� (TCS) - `temporal_coherence_score.py`

**����**: ������Ƶ�ж���ĳ��ֺ���ʧ�Ƿ����������ɺ�ʱ���߼���

**�����㷨**:
```python
def get_disappear_objects(tracking_result):
    # ���ͻȻ��ʧ�Ķ���
    for i in range(len(tracking_result) - 1):
        dict1 = tracking_result[i]
        dict2 = tracking_result[i + 1]
        
        # �ҳ���ʧ�ļ�
        disappeared_keys = set(dict1.keys()) - set(dict2.keys())
        
        # ������ʧԭ��
        for key in disappeared_keys:
            # ����Ƿ�ӱ�Ե��ʧ
            edge_vanish = is_edge_vanish(pred_tracks, pred_visibility, ...)
            # ����Ƿ����С����ʧ
            small_vanish = is_small_vanish(pred_tracks, pred_visibility, ...)
            # ����Ƿ�Ϊ������
            detect_error = is_vanish_detect_error(pred_tracks, pred_visibility, ...)
```

**����ջ**:
- Grounded-SAM-2: �߼�����ָ�͸���
- Co-Tracker: �˶��켣����
- ���������֤

**������׼**:
- �������/��ʧ�ĺ�����
- ��Ե��ʧ���
- �ߴ�仯����
- ������ʶ��

### 4. �˶�ƽ�������� (MSS) - `motion_smoothness_score.py`

**����**: ������Ƶ���˶��������������ԣ�����˶��е�αӰ�Ͳ������ԡ�

**�����㷨**:
```python
def get_artifacts_frames(scores, threshold=0.025):
    # ��������֡��ķ�������
    score_diffs = np.abs(np.diff(scores))
    
    # ʶ��������쳬����ֵ��֡
    artifact_indices = np.where(score_diffs > threshold)[0]
    
    # �����������֡
    artifacts_frames = np.unique(np.concatenate([artifact_indices, artifact_indices + 1]))
    
    return artifacts_frames

def set_threshold(camera_movement):
    # ��������˶�����������ֵ
    if camera_movement < 0.1:
        return 0.01
    elif 0.1 <= camera_movement < 0.3:
        return 0.015
    # ... ������ֵ����
```

**����ջ**:
- Q-Align: ��Ƶ��������ģ��
- �������ڷ���
- ����Ӧ��ֵ����

**��������**:
- ʹ��Q-Alignģ������ÿ֡����
- ��������˶����������ֵ
- ʶ�������쳣��֡����

### 5. ��ʶ��ѭ���� (CAS) - `commonsense_adherence_score.py`

**����**: ������Ƶ�����Ƿ��������ʶ����ʵ�߼���

**�����㷨**:
```python
def final_merge(eval_path, num_tasks, meta_info_path, method='prob'):
    # �ϲ�������̵��������
    for row in ans:
        video_id, prob, _, _, _, _ = row
        # �����Ȩ�ܷ�
        prob_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        total_score = np.sum(prob * prob_weights)
        item['commonsense_adherence_score'] = total_score
```

**����ջ**:
- VideoMAEv2: ��Ƶ���ģ��
- �ֲ�ʽ����
- ����̽���ϲ�

**����ά��**:
- ���������ѭ
- ��ʵ�߼�һ����
- ��ʶ������

## ����ģ�����

### Ԫ��Ϣ���� - `create_meta_info.py`

**����**: �����͹�����Ƶ������Ԫ��Ϣ�ļ���

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

### �������� - `calculate_score.py`

**����**: �����ά�ȵ�ƽ������������CSV���档

```python
def calculate_averages(json_file, output_csv):
    scores = {
        'perceptible_amplitude_socre': [],
        'object_integrity_score': [],
        'temporal_coherence_score': [],
        'motion_smoothness_score': [],
        'commonsense_adherence_score': []
    }
    
    # ����ƽ������
    averages = {key: mean(values) for key, values in scores.items() if values}
    total_score = mean(averages.values()) if averages else 0
```

### ��̬�������� - `pose_utils.py`

**����**: ����������̬�������Ժ�һ���ԡ�

**���Ĺ���**:
- `analyze_lengths_over_time()`: �������岿λ���ȱ仯
- `analyze_joint_angles()`: �����ؽڽǶȱ仯
- �쳣�������ּ���

## ��������

### ������������ - `evaluate.sh`

```bash
#!/bin/bash
VIDEO_DIR=$1
current_time=$(date "+%Y%m%d_%H%M%S")
META_INFO_PATH="./eval_results/${current_time}/results.json"

# 1. ����Ԫ��Ϣ
python bench_utils/create_meta_info.py -v $VIDEO_DIR -o $META_INFO_PATH

# 2. PAS ����
python perceptible_amplitude_score.py --meta_info_path $META_INFO_PATH

# 3. OIS ����
python object_integrity_score.py --meta-info-path $META_INFO_PATH

# 4. TCS ����
python temporal_coherence_score.py --meta_info_path $META_INFO_PATH

# 5. CAS ���� (�ֲ�ʽ)
torchrun --nproc_per_node=1 commonsense_adherence_score.py \
    --model vit_giant_patch14_224 \
    --data_set Commonsense-Adherence \
    --meta_info_path ${META_INFO_PATH}

# 6. MSS ���� (����PAS���)
python motion_smoothness_score.py --meta_info_path $META_INFO_PATH

# 7. �������ձ���
python bench_utils/calculate_score.py -i $META_INFO_PATH -o $META_INFO_DIR"/scores.csv"
```

## ������ϵ

### ��������
- **PyTorch**: ���ѧϰ���
- **OpenCV**: ��Ƶ����
- **NumPy**: ��ֵ����
- **PIL**: ͼ����

### ������ģ��
- **GroundedDINO**: ������
- **SAM/SAM2**: ����ָ�
- **Co-Tracker**: �˶�����
- **MMPose**: ��̬����
- **Q-Align**: ��������
- **VideoMAEv2**: ��Ƶ���

### ��װҪ��
```bash
# ��������
conda create -n VMBench python=3.10
pip install torch==2.5.1 torchvision==0.20.1

# ��ģ�鰲װ
cd Grounded-Segment-Anything && pip install -e .
cd ../Grounded-SAM-2 && pip install -e .
cd ../mmpose && pip install -e .
cd ../Q-Align && pip install -e .
cd ../VideoMAEv2 && pip install -r requirements.txt
```

## ʹ��ʾ��

### 1. ��Ƶ����
```python
# ʹ��ʾ��ģ��������Ƶ
from diffusers import CogVideoXPipeline

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b")
video = pipe(prompt="A green turtle swims alongside dolphins", num_frames=49)
```

### 2. ����ִ��
```bash
# ׼����Ƶ�ļ�
python sample_video_demo.py --prompt_path ./prompts/prompts.json --save_dir ./eval_results/videos

# ������������
bash evaluate.sh ./eval_results/videos
```

### 3. �������
```python
# ��ȡ�������
import json
with open('./eval_results/20250101_120000/scores.csv', 'r') as f:
    results = json.load(f)
    
# ������ά�ȷ���
print(f"PAS: {results['perceptible_amplitude_score']}")
print(f"OIS: {results['object_integrity_score']}")
print(f"TCS: {results['temporal_coherence_score']}")
print(f"MSS: {results['motion_smoothness_score']}")
print(f"CAS: {results['commonsense_adherence_score']}")
```

## �����ص�

### 1. ��ά������
- **PAS**: �˶����ȸ�֪
- **OIS**: ����������
- **TCS**: ʱ��һ����
- **MSS**: �˶�ƽ����
- **CAS**: ��ʶ��ѭ

### 2. ��֪����
- ���������֪��������׼
- 35.3%��Spearman���������
- ����ƫ����֤����

### 3. �����Ƚ���
- �������µļ�����Ӿ�ģ��
- �ֲ�ʽ����֧��
- ����Ӧ��ֵ����
- ����̲��д���

### 4. ����չ��
- ģ�黯���
- ��������µ�����ά��
- ֧���Զ���������׼

## ����ָ��

### ����Ч��
����CogVideoX-5Bģ�͵�1050����Ƶ����ʱ�䣺

| ����ά�� | ��ʱ | ˵�� |
|---------|------|------|
| PAS | 45���� | ��������˶����� |
| OIS | 30���� | ��̬���ƺͷ��� |
| TCS | 2Сʱ | ������ٺ�һ���Է��� |
| MSS | 2.5Сʱ | ����������αӰ��� |
| CAS | 1Сʱ | ��ʶ��ѭ���� |
| **�ܼ�** | **6Сʱ45����** | **������������** |

### ģ�����ܶԱ�
| ģ�� | ƽ���� | PAS | OIS | TCS | MSS | CAS |
|------|--------|-----|-----|-----|-----|-----|
| OpenSora-v1.2 | 51.6 | 31.2 | 61.9 | 73.0 | 3.4 | 88.5 |
| Mochi 1 | 53.2 | 37.7 | 62.0 | 68.6 | 14.4 | 83.6 |
| OpenSora-Plan-v1.3.0 | 58.9 | 39.3 | 76.0 | 78.6 | 6.0 | 94.7 |
| CogVideoX-5B | 60.6 | 50.6 | 61.6 | 75.4 | 24.6 | 91.0 |
| HunyuanVideo | 63.4 | 51.9 | 81.6 | 65.8 | 26.1 | 96.3 |
| Wan2.1 | **78.4** | **62.8** | **84.2** | 66.0 | 17.9 | **97.8** |

## �ܽ�

VMBench��һ��ȫ�����Ƶ�˶�����������׼���Թ��ߣ�ͨ���������ά���ṩ��ϸ���ȵ��������������֪��������������Ƚ��ļ���ջʹ���Ϊ��Ƶ����ģ����������Ҫ���ߡ�ģ�黯�ļܹ��ͷḻ�Ĺ���ʹ����ʺ��о�ʹ�ã�Ҳ�ʺ�ʵ��Ӧ�ò���

����Ŀ�ĳɹ����ڣ�
1. **ȫ�����������**: ���˶����ȵ���ʶ��ѭ��ȫ��λ����
2. **��֪�������**: ���������֪��������׼
3. **�����Ƚ���**: �������µļ�����Ӿ������ѧϰ����
4. **ʵ����ǿ**: �ṩ�������������̺͹�����
5. **����չ��**: ģ�黯��Ʊ�����չ��ά��
