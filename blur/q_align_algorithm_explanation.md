# Q-Align�㷨ʵ���߼����

## 1. �㷨����

Q-Align��һ�����ڶ�ģ̬������ģ�͵���Ƶ���������㷨����Ҫ����������Ƶ�ĸ�֪��������������Ӿ�������������ģ�ͺ�ƫ��ѧϰ���ơ�

## 2. ���ļܹ�

### 2.1 ģ�����

```
Q-Align = �Ӿ������� + �Ӿ������� + ����ģ�� + ƫ��ѧϰͷ
```

- **�Ӿ�������**: ����CLIP���Ӿ�������ȡ
- **�Ӿ�������**: ���Ӿ�����ת��Ϊ����ģ�Ϳ����ı�ʾ
- **����ģ��**: ����LLaMA���ı����
- **ƫ��ѧϰͷ**: ����������ת��Ϊƫ��ѡ��

### 2.2 ����ջ

- **����ģ��**: mPLUG-Owl2 (��ģ̬������ģ��)
- **�Ӿ�����**: CLIP ViT
- **����ģ��**: LLaMA-2
- **ѵ�����**: Transformers + LoRA΢��

## 3. ����Ҫ��

### 3.1 �����ʽ

```python
# �������ݽṹ
input_data = {
    'video_frames': List[List[PIL.Image]],  # ��Ƶ֡����
    'prompt': str,                         # ������ʾ
    'window_size': int                     # �������ڴ�С
}
```

### 3.2 ��������Ҫ��

#### 3.2.1 ��Ƶ֡����
```python
# ��Ƶ֡Ҫ��
video_frames = [
    [frame1, frame2, frame3, ...],  # ��һ��ʱ�䴰��
    [frame2, frame3, frame4, ...],  # �ڶ���ʱ�䴰��
    ...
]

# ֡��ʽҪ��
- ����: PIL.Image
- ��ɫ�ռ�: RGB
- �ߴ�: ���⣨���Զ�������
- ����: ����window_sizeȷ��
```

#### 3.2.2 ������ʾ
```python
# Ĭ����ʾ
prompt = "USER: How would you rate the quality of this video?\n<|image|>\nASSISTANT: The quality of the video is"

# ��ʾ�ṹ
- USER: �û�����
- <|image|>: ͼ��ռλ��
- ASSISTANT: ģ�ͻش�
```

#### 3.2.3 �������ڲ���
```python
window_size = 3  # ÿ�����ڰ���3֡
# ���ڻ�����ʽ��
# ֡1: [frame1, frame1, frame1]  # ��ʼ֡�ظ�
# ֡2: [frame1, frame2, frame3]  # ��������
# ֡3: [frame2, frame3, frame4]  # ��������
# ...
```

## 4. �㷨ʵ������

### 4.1 ��ʼ���׶�

```python
class QAlignVideoScorer(nn.Module):
    def __init__(self, pretrained="q-future/one-align", device="cuda:0"):
        # 1. ����Ԥѵ��ģ��
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained, None, "mplug_owl2", device=device
        )
        
        # 2. ����������ʾ
        prompt = "USER: How would you rate the quality of this video?\n<|image|>\nASSISTANT: The quality of the video is"
        
        # 3. ���������ȼ���Ȩ��
        self.preferential_ids_ = [id_[1] for id_ in tokenizer([
            "excellent", "good", "fair", "poor", "bad"
        ])["input_ids"]]
        
        self.weight_tensor = torch.Tensor([1, 0.75, 0.5, 0.25, 0.]).half()
        
        # 4. Ԥ��������
        self.input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)
```

### 4.2 ��ƵԤ����

```python
def load_video_sliding_window(video_file, window_size=5):
    """
    �������ڷ�ʽ��ȡ��Ƶ
    
    Args:
        video_file: ��Ƶ�ļ�·��
        window_size: ���ڴ�С
    
    Returns:
        frames_by_group: ��������֯��֡����
    """
    vr = VideoReader(video_file)
    total_frames = len(vr)
    frames_by_group = []
    
    # ���㴰����չ
    left_extend = (window_size - 1) // 2
    right_extend = window_size - 1 - left_extend
    
    for current_frame in range(total_frames):
        # ���㴰�ڷ�Χ
        start_frame = max(0, current_frame - left_extend)
        end_frame = min(total_frames, current_frame + right_extend + 1)
        
        frame_indices = list(range(start_frame, end_frame))
        
        # ��䲻���֡
        while len(frame_indices) < window_size:
            if start_frame == 0:
                frame_indices.append(frame_indices[-1])  # �ظ����һ֡
            else:
                frame_indices.insert(0, frame_indices[0])  # �ظ���һ֡
        
        # ��ȡ֡����
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # ת��ΪPILͼ��
        if current_frame < left_extend:
            frames_by_group.append([Image.fromarray(frames[0])] * window_size)
        else:
            frames_by_group.append([Image.fromarray(frame) for frame in frames])
    
    return frames_by_group
```

### 4.3 ͼ��Ԥ����

```python
def expand2square(self, pil_img, background_color):
    """
    ��ͼ����չΪ������
    
    Args:
        pil_img: PILͼ��
        background_color: ������ɫ
    
    Returns:
        ������PILͼ��
    """
    width, height = pil_img.size
    
    if width == height:
        return pil_img
    elif width > height:
        # �߶�С�ڿ�ȣ���������ӱ���
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        # ���С�ڸ߶ȣ���������ӱ���
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
```

### 4.4 ǰ������

```python
def forward(self, video: List[List[Image.Image]]):
    """
    ǰ���������
    
    Args:
        video: ��Ƶ֡���� [batch_size, window_size, PIL.Image]
    
    Returns:
        output_logits: ԭʼlogits
        softmax_probs: ��������
        weighted_scores: ��Ȩ����
    """
    # 1. ͼ��Ԥ����
    video = [[self.expand2square(
        frame, 
        tuple(int(x*255) for x in self.image_processor.image_mean)
    ) for frame in vid] for vid in video]
    
    with torch.inference_mode():
        # 2. ͼ�����
        video_tensors = [
            self.image_processor.preprocess(vid, return_tensors="pt")["pixel_values"]
            .half().to(self.model.device) 
            for vid in video
        ]
        
        # 3. �ı�����
        input_tensors = self.input_ids.repeat(len(video_tensors), 1)
        
        # 4. ��ģ̬����
        output = self.model(input_tensors, images=video_tensors)
        
        # 5. ��ȡ��������logits
        output_logits = output["logits"][:, -1, self.preferential_ids_]
        
        # 6. ������ʷֲ�
        softmax_probs = torch.softmax(output_logits, -1)
        
        # 7. �����Ȩ����
        weighted_scores = softmax_probs @ self.weight_tensor
        
        return output_logits, softmax_probs, weighted_scores
```

## 5. ������������

### 5.1 ƫ��ѧϰ

```python
# �����ȼ�����
quality_levels = ["excellent", "good", "fair", "poor", "bad"]

# ��ӦȨ��
weights = [1.0, 0.75, 0.5, 0.25, 0.0]

# ���շ�������
final_score = ��(probability_i �� weight_i)
```

### 5.2 ��������

- **1.0**: ��������
- **0.75**: ��������  
- **0.5**: һ������
- **0.25**: �ϲ�����
- **0.0**: �ܲ�����

## 6. ģ�����Ӧ��

### 6.1 ������������

```python
def get_artifacts_frames(scores, threshold=0.025):
    """
    �������������������쳣֡
    
    Args:
        scores: ������������
        threshold: �����ֵ
    
    Returns:
        artifact_indices: �쳣֡����
    """
    # ��������֡��������
    score_diffs = np.abs(np.diff(scores))
    
    # ʶ�𳬹���ֵ��֡
    artifact_indices = np.where(score_diffs > threshold)[0]
    
    # �����쳣֡��������ǰ֡����һ֡��
    artifacts_frames = np.unique(np.concatenate([
        artifact_indices, 
        artifact_indices + 1
    ]))
    
    return artifacts_frames
```

### 6.2 ����Ӧ��ֵ

```python
def set_threshold(camera_movement):
    """
    ��������˶�������������Ӧ��ֵ
    
    Args:
        camera_movement: ����˶����� (0-1)
    
    Returns:
        threshold: �����ֵ
    """
    if camera_movement is None:
        return 0.01
    elif camera_movement < 0.1:
        return 0.01      # ��̬����������ֵ
    elif camera_movement < 0.3:
        return 0.015     # ��΢�˶����е���ֵ
    elif camera_movement < 0.5:
        return 0.025     # �е��˶����ϸ���ֵ
    else:
        return 0.03      # �����˶�������ֵ
```

## 7. �����Ż�

### 7.1 �ڴ��Ż�

```python
# ʹ�ð뾫������
model.half()

# ����ģʽ
with torch.inference_mode():
    output = model(inputs)
```

### 7.2 �������Ż�

```python
# ������������Ƶ����
batch_size = 8
for i in range(0, len(video_windows), batch_size):
    batch = video_windows[i:i+batch_size]
    scores = model(batch)
```

## 8. ʹ��ʾ��

### 8.1 ����ʹ��

```python
from motion_smoothness_score import QAlignVideoScorer, load_video_sliding_window

# ��ʼ��ģ��
scorer = QAlignVideoScorer(pretrained="q-future/one-align", device="cuda:0")

# ������Ƶ
video_frames = load_video_sliding_window("video.mp4", window_size=3)

# ��������
_, _, scores = scorer(video_frames)
print(f"��������: {scores}")
```

### 8.2 ģ�����

```python
# ���ģ��֡
blur_frames = get_artifacts_frames(scores, threshold=0.025)
print(f"��⵽ģ��֡: {blur_frames}")

# ����MSS����
mss_score = 1 - len(blur_frames) / len(scores)
print(f"MSS����: {mss_score}")
```

## 9. �㷨����

1. **��֪����**: ���������֪����������
2. **��ģ̬���**: ����Ӿ���������Ϣ
3. **����Ӧ���**: ���ݳ�����̬������ֵ
4. **��Ч����**: �Ż���ģ�ͼܹ�����������
5. **�ɽ�����**: �ṩ��ϸ�������������

## 10. ������

1. **������Դ**: ��ҪGPU����
2. **ģ�ʹ�С**: Ԥѵ��ģ�ͽϴ�
3. **ʵʱ��**: �����ٶ���Խ���
4. **������**: ���ض���������Ӧ������

## 11. �Ľ�����

1. **ģ��ѹ��**: ����ģ�ʹ�С�ͼ�����
2. **ʵʱ�Ż�**: ��������ٶ�
3. **��߶ȼ��**: ֧�ֲ�ͬ�ֱ��ʵ���Ƶ
4. **������Ӧ**: ����ض�������΢��
