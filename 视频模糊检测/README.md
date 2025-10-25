# ��Ƶģ�����ϵͳ

����VMBench����Ƶģ�����ϵͳ����Ҫʹ��MSS (�˶�ƽ��������) �� PAS (�ɸ�֪��������) ����ģ����⡣

## ��������

- **����VMBench**: ����VMBench��MSS��PAS����������ģ�����
- **����Ӧ��ֵ**: ��������˶������Զ����������ֵ
- **�༶���**: ֧����΢���еȡ�����ģ��������
- **���ӻ�֧��**: �ṩ�ḻ�Ŀ��ӻ����
- **��������**: ֧��������Ƶ���
- **�������**: ֧�ֶ���Ԥ������

## ϵͳ�ܹ�

```
��Ƶģ�����/
������ blur_detection_pipeline.py    # ��������ܵ�
������ simple_blur_detector.py       # �򻯰�����
������ blur_visualization.py         # ���ӻ�����
������ config.py                     # ���ù���
������ run_blur_detection.py         # ���нű�
������ README.md                     # ˵���ĵ�
```

## ��װҪ��

### ��������
```bash
pip install torch torchvision
pip install opencv-python
pip install numpy matplotlib seaborn
pip install pillow tqdm
```

### VMBench����
ȷ���Ѱ�װVMBench������������������
- Q-Alignģ��
- GroundingDINO
- SAM (Segment Anything Model)
- Co-Tracker

## ���ٿ�ʼ

### 1. ����Ƶ���

```bash
# ʹ�ü򻯰��������Ƽ���
python run_blur_detection.py --video_path /path/to/video.mp4

# ʹ������������
python run_blur_detection.py --video_path /path/to/video.mp4 --use_simple=False
```

### 2. �������

```bash
# ���������ƵĿ¼
python run_blur_detection.py --video_dir /path/to/video/directory

# ָ�����Ŀ¼
python run_blur_detection.py --video_dir /path/to/videos --output_dir ./results
```

### 3. ����Ԥ��

```bash
# ���ټ�⣨�ٶ����ȣ�
python run_blur_detection.py --video_path /path/to/video.mp4 --config_preset fast

# ��ȷ��⣨�������ȣ�
python run_blur_detection.py --video_path /path/to/video.mp4 --config_preset accurate

# ƽ���⣨Ĭ�ϣ�
python run_blur_detection.py --video_path /path/to/video.mp4 --config_preset balanced
```

## ��ϸʹ��

### 1. �򻯰�����

```python
from simple_blur_detector import SimpleBlurDetector

# ��ʼ�������
detector = SimpleBlurDetector(device="cuda:0")

# ��ⵥ����Ƶ
result = detector.detect_blur("video.mp4")

# �������
results = detector.batch_detect("/path/to/videos", "./results")
```

### 2. ����������

```python
from blur_detection_pipeline import BlurDetectionPipeline

# ��ʼ�������
detector = BlurDetectionPipeline(device="cuda:0")

# ��ⵥ����Ƶ
result = detector.detect_blur_in_video("video.mp4", subject_noun="person")

# �������
results = detector.batch_detect_blur("/path/to/videos", "./results")
```

### 3. ���ӻ�����

```python
from blur_visualization import BlurVisualization

# ��ʼ�����ӻ�����
visualizer = BlurVisualization("./visualizations")

# ���ӻ���������
visualizer.visualize_quality_scores(video_path, quality_scores, blur_frames, threshold)

# ���ӻ�ģ��֡
visualizer.visualize_blur_frames(video_path, blur_frames)

# ���ɼ�ⱨ��
visualizer.create_detection_report(result)
```

## ����˵��

### ������

```python
from config import BlurDetectionConfig

config = BlurDetectionConfig()

# ���¼�����
config.update_detection_param('window_size', 5)
config.update_detection_param('confidence_threshold', 0.8)

# ����ģ��·��
config.update_model_path('q_align_model', '/path/to/q-align')
```

### Ԥ������

- **fast**: ���ټ�⣬�ʺ�ʵʱӦ��
- **accurate**: ��ȷ��⣬�ʺ�����Ҫ��ߵĳ���
- **balanced**: ƽ���⣬Ĭ���Ƽ�����

## ������

### �������ʽ

```json
{
  "video_path": "/path/to/video.mp4",
  "video_name": "video.mp4",
  "blur_detected": true,
  "confidence": 0.85,
  "blur_severity": "�е�ģ��",
  "blur_ratio": 0.15,
  "blur_frame_count": 15,
  "total_frames": 100,
  "avg_quality": 0.75,
  "max_quality_drop": 0.12,
  "threshold": 0.025,
  "blur_frames": [10, 15, 20, 25, 30],
  "recommendations": [
    "����ʹ���ȶ���",
    "���¼��֡��",
    "ȷ���������"
  ]
}
```

### ���ӻ����

- **���������仯ͼ**: ��ʾ��Ƶ����������ʱ��ı仯
- **ģ��֡չʾ**: չʾ��⵽��ģ��֡
- **��ⱨ��**: �ۺϵļ��������
- **����ͳ��**: ��������ͳ�ƽ��

## �㷨ԭ��

### 1. MSS������ (��Ҫ���)

����Q-Alignģ��������Ƶ������
1. ʹ�û���������ȡ��Ƶ֡
2. ͨ��Q-Alignģ�ͼ���ÿ֡��������
3. ��������֡��������������
4. ��������˶�������������Ӧ��ֵ
5. ʶ�������쳣�½���֡��Ϊģ��֡

### 2. PAS������ (������֤)

�����˶�������֤ģ����
1. ʹ��GroundingDINO����������
2. ʹ��SAM���ɾ�ȷ��������
3. ʹ��Co-Tracker�����˶��켣
4. �����һ���˶�����
5. ģ���ᵼ���˶����ٲ�׼ȷ���˶������쳣��

### 3. �ۺ��ж�

```python
# �ۺ����Ŷȼ���
confidence = mss_score * 0.8 + pas_score * 0.2

# ģ������ж�
blur_detected = (
    len(blur_frames) > 0 and 
    confidence < confidence_threshold
)
```

## �����Ż�

### 1. ģ���Ż�
- ʹ�ü򻯰��������ټ�����
- �����������ڴ�Сƽ���ٶȺ;���
- ʹ��GPU���ټ���

### 2. �������Ż�
- ���д�������Ƶ
- �ڴ�����Ż�
- ����������

### 3. ���õ���
- ���ݳ���ѡ����ʵ�����Ԥ��
- ���������ֵƽ���󱨺�©��
- �Ż����ӻ�����

## �����ų�

### ��������

1. **ģ�ͼ���ʧ��**
   ```
   �������: ���ģ���ļ�·����ȷ������������Ԥѵ��ģ��
   ```

2. **CUDA�ڴ治��**
   ```
   �������: ����batch_size����ʹ��CPUģʽ
   ```

3. **��⾫�Ȳ���**
   ```
   �������: ʹ��accurate���ã�����������ֵ
   ```

4. **�����ٶ���**
   ```
   �������: ʹ��fast���ã��������Ƶ�ֱ���
   ```

### ����ģʽ

```bash
# ������ϸ��־
python run_blur_detection.py --video_path video.mp4 --verbose

# �������ӻ����ٴ���
python run_blur_detection.py --video_path video.mp4 --no_visualization
```

## ��չ����

### ����µļ���㷨

```python
class CustomBlurDetector(SimpleBlurDetector):
    def detect_blur(self, video_path):
        # ʵ���Զ������߼�
        pass
```

### �Զ�����ӻ�

```python
class CustomVisualization(BlurVisualization):
    def custom_visualization(self, results):
        # ʵ���Զ�����ӻ�
        pass
```

## ���֤

����Ŀ����VMBench��������ѭ��ͬ�����֤���

## ����

��ӭ�ύIssue��Pull Request���Ľ�����Ŀ��

## ������־

- v1.0.0: ��ʼ�汾��֧�ֻ���ģ�����
- v1.1.0: ��ӿ��ӻ�����
- v1.2.0: �Ż����ܣ������������
