# CASע����Ȩ�ؿ��ӻ�ʹ��ָ��

## ����

��ָ�Ͻ������ʹ����ǿ��CAS����ϵͳ����ע����Ȩ�ؿ��ӻ������������Ƶ����Щ������ڳ�ʶΥ����

## ��������

- ? **ע����Ȩ����ȡ**: ��VideoMAEv2ģ����ȡע����Ȩ��
- ? **����ͼ���ӻ�**: ����ע��������ͼ��ʾ�쳣����
- ? **��Ƶ֡��ע**: ����Ƶ֡�ϵ���ע������Ϣ
- ? **�쳣���**: �Զ�ʶ���CAS���ֵ���Ƶ
- ? **��������**: ֧��������Ƶ�����Ϳ��ӻ�
- ? **��ϸ����**: ���������ķ�������

## ��װ����

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib seaborn
pip install numpy pandas
```

## ���ٿ�ʼ

### 1. ����Ƶ����

```python
from enhanced_commonsense_adherence_score import EnhancedCASScorer
import torch

# ��ʼ������
args = {
    'model': 'vit_base_patch16_224',
    'input_size': 224,
    'num_frames': 16,
    'tubelet_size': 2,
    'device': 'cuda',
    'enable_visualization': True,
    'visualization_threshold': 0.5
}

# ����ģ��
model = create_model(args['model'], ...)
model.to(args['device'])

# ����CAS������
cas_scorer = EnhancedCASScorer(args, model, args['device'])

# ����������Ƶ
result = cas_scorer.evaluate_with_visualization('path/to/video.mp4')
print(f"CAS����: {result['cas_score']:.3f}")
```

### 2. ��������

```python
# ����������ƵĿ¼
results = cas_scorer.batch_evaluate_with_visualization(
    'path/to/video/directory',
    output_dir='./visualization_results'
)
```

### 3. ������ʹ��

```bash
# ����Ƶ����
python enhanced_commonsense_adherence_score.py \
    --video_path path/to/video.mp4 \
    --enable_visualization \
    --output_dir ./results

# ��������
python enhanced_commonsense_adherence_score.py \
    --video_dir path/to/videos \
    --enable_visualization \
    --output_dir ./batch_results
```

## ������

### 1. ���ӻ��ļ�

- `attention_heatmap.png`: ע��������ͼ
- `frame_XXX_attention.jpg`: ��ע������ע����Ƶ֡
- `attention_report.txt`: ��ϸ��������

### 2. �������

- `cas_visualization_results.json`: �����������
- `cas_summary.csv`: ����ժҪ
- `statistics_report.txt`: ͳ�Ʊ���

## ����˵��

### ���Ĳ���

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `--enable_visualization` | bool | False | ����ע�������ӻ� |
| `--visualization_threshold` | float | 0.5 | ���ӻ�������ֵ |
| `--output_dir` | str | ./cas_visualization_results | ���Ŀ¼ |
| `--video_path` | str | None | ����Ƶ·�� |
| `--video_dir` | str | None | ��ƵĿ¼·�� |

### ģ�Ͳ���

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `--model` | str | vit_base_patch16_224 | ģ������ |
| `--input_size` | int | 224 | ����ͼ��ߴ� |
| `--num_frames` | int | 16 | ��Ƶ֡�� |
| `--tubelet_size` | int | 2 | ʱ��tubelet��С |

## ʹ��ʾ��

### ʾ��1����ⷿ�������̵����

```python
# ���������쳣�����������Ƶ
result = cas_scorer.evaluate_with_visualization('eaves_frozen_raindrops.mp4')

if result['cas_score'] < 0.5:
    print("��⵽��CAS���֣����ܴ��������쳣")
    print("��鿴ע��������ͼ�˽��쳣����")
    
    # �鿴���ӻ����
    if result['visualization']:
        print(f"���ӻ����������: {result['visualization']['output_dir']}")
```

### ʾ��2����������쳣��Ƶ

```python
# ����������ƵĿ¼
results = cas_scorer.batch_evaluate_with_visualization('./test_videos')

# �������
low_score_videos = [r for r in results if r['cas_score'] < 0.5]
print(f"��⵽ {len(low_score_videos)} ���쳣��Ƶ")

for video in low_score_videos:
    print(f"�쳣��Ƶ: {video['video_path']}, ����: {video['cas_score']:.3f}")
```

## ���ӻ�������

### 1. ע��������ͼ

- **��ɫ����**: ��ע�������������쳣����
- **��ɫ����**: ��ע��������������
- **��ɫǿ��**: ��ʾע����Ȩ�ش�С

### 2. �쳣���

- **CAS���� < 0.3**: ����Υ����ʶ
- **CAS���� 0.3-0.5**: ����Υ����ʶ
- **CAS���� > 0.5**: �������ϳ�ʶ

### 3. ��������

�������������Ϣ��
- CAS����ͳ��
- ע�����ֲ�����
- ��ע��������ʶ��
- �쳣ԭ�����

## �����ų�

### ��������

1. **ģ�ͼ���ʧ��**
   ```bash
   # ȷ��ģ��·����ȷ
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **��Ƶ����ʧ��**
   ```bash
   # �����Ƶ��ʽ��·��
   ffmpeg -i video.mp4 -f null -
   ```

3. **�ڴ治��**
   ```python
   # �����������С
   args.batch_size = 1
   ```

### �����Ż�

1. **GPU����**: ȷ��ʹ��CUDA�豸
2. **������**: ���������������С
3. **�ڴ����**: ��ʱ�ͷŲ���Ҫ��tensor

## �߼�����

### 1. �Զ���ע��������

```python
# ��ȡ�ض����ע����Ȩ��
attention_weights, layer_names, outputs = visualizer.extract_attention_weights(video_tensor)

# �����ض����ע����
layer_attention = attention_weights[-1]  # ���һ��
```

### 2. ע����Ȩ�ط���

```python
# ����ע����ͳ��
attention_stats = {
    'mean': np.mean(attention_heatmap),
    'std': np.std(attention_heatmap),
    'max': np.max(attention_heatmap),
    'min': np.min(attention_heatmap)
}
```

## ����ԭ��

### 1. ע��������

VideoMAEv2ʹ�ö�ͷ��ע�������ƣ�
- ÿ��patch��Ӧ��Ƶ�е�һ��ʱ������
- ע����Ȩ�ر�ʾ��ͬ����֮��������
- �쳣����ͨ�������쳣��ע����ģʽ

### 2. ���ӻ�����

- **Rollout����**: �ۺ϶��ע����Ȩ��
- **����ͼ����**: ��ע����Ȩ��ӳ�䵽�ռ�λ��
- **��ɫ����**: ʹ����ɫ��ʾע����ǿ��

### 3. �쳣���

- **��ֵ���**: ����CAS������ֵ
- **ע��������**: ����ע�����ֲ�ģʽ
- **�����ע**: ��ע��ע��������

## ��չ����

### 1. �Զ�����ӻ�

```python
# �Զ�����ɫӳ��
def custom_colormap(attention_map):
    # ʵ���Զ�����ɫӳ���߼�
    pass
```

### 2. ��ģ̬����

```python
# �������VMBenchָ��
def multi_modal_analysis(video_path):
    cas_result = cas_scorer.evaluate_with_visualization(video_path)
    # �������ָ����з���
    return combined_analysis
```

## �ܽ�

CASע����Ȩ�ؿ��ӻ�ϵͳ�ṩ��ǿ�����Ƶ�쳣���Ϳ��ӻ����ܣ��ܹ���

1. **�Զ����**: ʶ��Υ����ʶ����Ƶ����
2. **��ȷ��λ**: ָ��������쳣����
3. **��ϸ����**: �ṩ����ķ�������
4. **��������**: ֧�ִ��ģ��Ƶ����

ͨ��ʹ�ñ�ϵͳ�������Ը��õ������Ƶ����ģ�͵���Ϊ��ʶ��ͽ����ʶΥ�����⡣
