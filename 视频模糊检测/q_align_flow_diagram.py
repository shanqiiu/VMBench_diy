# -*- coding: utf-8 -*-
"""
Q-Align�㷨����ͼ������
���ڿ��ӻ�Q-Align�㷨�Ĺ�������
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_q_align_flow_diagram():
    """����Q-Align�㷨����ͼ"""
    
    # ����ͼ��
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # ������ɫ
    colors = {
        'input': '#E3F2FD',
        'preprocess': '#F3E5F5',
        'vision': '#E8F5E8',
        'language': '#FFF3E0',
        'output': '#FFEBEE',
        'arrow': '#666666'
    }
    
    # 1. ����׶�
    input_box = FancyBboxPatch(
        (0.5, 10.5), 2, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['input'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(1.5, 11, '��Ƶ����\n(Video Input)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 2. ��ƵԤ����
    preprocess_box = FancyBboxPatch(
        (3.5, 10.5), 2, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['preprocess'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(preprocess_box)
    ax.text(4.5, 11, '��������\n(Sliding Window)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 3. ͼ��Ԥ����
    img_preprocess_box = FancyBboxPatch(
        (6.5, 10.5), 2, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['preprocess'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(img_preprocess_box)
    ax.text(7.5, 11, 'ͼ��Ԥ����\n(Image Preprocess)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 4. �Ӿ�������
    vision_box = FancyBboxPatch(
        (1, 8.5), 2.5, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['vision'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(vision_box)
    ax.text(2.25, 9, 'CLIP�Ӿ�������\n(CLIP Vision Encoder)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 5. �Ӿ�������
    visual_abstractor_box = FancyBboxPatch(
        (4, 8.5), 2.5, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['vision'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(visual_abstractor_box)
    ax.text(5.25, 9, '�Ӿ�������\n(Visual Abstractor)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 6. ����ģ��
    language_box = FancyBboxPatch(
        (7, 8.5), 2.5, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['language'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(language_box)
    ax.text(8.25, 9, 'LLaMA����ģ��\n(LLaMA Language Model)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 7. ƫ��ѧϰͷ
    preference_box = FancyBboxPatch(
        (2, 6.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(preference_box)
    ax.text(3.5, 7, 'ƫ��ѧϰͷ\n(Preference Learning Head)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 8. ��������
    quality_box = FancyBboxPatch(
        (6, 6.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(quality_box)
    ax.text(7.5, 7, '��������\n(Quality Score)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 9. ģ�����
    blur_detection_box = FancyBboxPatch(
        (1, 4.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(blur_detection_box)
    ax.text(2.5, 5, 'ģ�����\n(Blur Detection)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 10. ������
    output_box = FancyBboxPatch(
        (5, 4.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(6.5, 5, '�����\n(Detection Results)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ��Ӽ�ͷ����
    arrows = [
        # ���뵽Ԥ����
        ((2.5, 10.5), (3.5, 10.5)),
        # Ԥ����ͼ��Ԥ����
        ((5.5, 10.5), (6.5, 10.5)),
        # ͼ��Ԥ�����Ӿ�������
        ((7.5, 10), (2.25, 9.5)),
        # �Ӿ����������Ӿ�������
        ((3.5, 9), (4, 9)),
        # �Ӿ�������������ģ��
        ((6.5, 9), (7, 9)),
        # ����ģ�͵�ƫ��ѧϰͷ
        ((8.25, 8), (3.5, 7.5)),
        # ƫ��ѧϰͷ����������
        ((5, 7), (6, 7)),
        # ����������ģ�����
        ((7.5, 6), (2.5, 5.5)),
        # ģ����⵽������
        ((4, 5), (5, 5))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
        ax.add_patch(arrow)
    
    # �����ϸ˵��
    details = [
        (0.5, 9.5, "? ��Ƶ֡����\n? PIL.Image��ʽ\n? RGB��ɫ�ռ�"),
        (3.5, 9.5, "? ��������\n? ֡���\n? ʱ�����"),
        (6.5, 9.5, "? ��������չ\n? �ߴ��׼��\n? ��ɫ��һ��"),
        (1, 8, "? CLIP ViT\n? �Ӿ�������ȡ\n? ��߶ȱ���"),
        (4, 8, "? ��������\n? ���Զ���\n? ��ʾѧϰ"),
        (7, 8, "? LLaMA-2\n? �ı����\n? ��ģ̬�ں�"),
        (2, 6, "? �����ȼ�\n? ƫ��ѧϰ\n? Ȩ�ؼ���"),
        (6, 6, "? 0-1����\n? ���ʷֲ�\n? ���Ŷ�"),
        (1, 3.5, "? ��������\n? ��ֵ���\n? �쳣ʶ��"),
        (5, 3.5, "? ģ��֡λ��\n? ���س̶�\n? �Ľ�����")
    ]
    
    for x, y, text in details:
        ax.text(x, y, text, fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # ��ӱ���
    ax.text(5, 11.8, 'Q-Align�㷨����ͼ', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # ���ͼ��
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['input'], label='����׶�'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['preprocess'], label='Ԥ����׶�'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['vision'], label='�Ӿ�����'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['language'], label='���Դ���'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['output'], label='����׶�')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def create_input_requirements_diagram():
    """��������Ҫ��ͼ��"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # ����
    ax.text(5, 7.5, 'Q-Align����Ҫ�����', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # �����ʽ
    input_format_box = FancyBboxPatch(
        (0.5, 6), 4, 1,
        boxstyle="round,pad=0.1",
        facecolor='#E3F2FD',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_format_box)
    ax.text(2.5, 6.5, '�����ʽ (Input Format)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # ��Ƶ֡Ҫ��
    video_requirements = """
��Ƶ֡Ҫ�� (Video Frame Requirements):
? ����: PIL.Image
? ��ɫ�ռ�: RGB
? �ߴ�: ���⣨�Զ�������
? ����: ����window_sizeȷ��
? ��ʽ: List[List[PIL.Image]]
    """
    
    ax.text(0.5, 5, video_requirements, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.8))
    
    # ������ʾ
    prompt_requirements = """
������ʾ (Evaluation Prompt):
? Ĭ��: "USER: How would you rate the quality of this video?\\n<|image|>\\nASSISTANT: The quality of the video is"
? �ṹ: USER + <|image|> + ASSISTANT
? ռλ��: <|image|> ����ͼ�����
    """
    
    ax.text(5.5, 5, prompt_requirements, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.8))
    
    # ��������
    window_requirements = """
�������� (Sliding Window):
? window_size: 3-5֡
? ������: �߽�֡�ظ�
? ʱ�����: ����֡����
? �ص�: ���ڴ����ص�
    """
    
    ax.text(0.5, 3, window_requirements, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.8))
    
    # �����ȼ�
    quality_levels = """
�����ȼ� (Quality Levels):
? excellent: 1.0 (����)
? good: 0.75 (����)
? fair: 0.5 (һ��)
? poor: 0.25 (�ϲ�)
? bad: 0.0 (�ܲ�)
    """
    
    ax.text(5.5, 3, quality_levels, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.8))
    
    # ����ʾ��
    code_example = """
����ʾ�� (Code Example):
```python
# ��ʼ��ģ��
scorer = QAlignVideoScorer(device="cuda:0")

# ������Ƶ
video_frames = load_video_sliding_window("video.mp4", window_size=3)

# ��������
_, _, scores = scorer(video_frames)
```
    """
    
    ax.text(0.5, 1, code_example, fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F5E8', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """������"""
    print("����Q-Align�㷨����ͼ...")
    
    # ��������ͼ
    fig1 = create_q_align_flow_diagram()
    fig1.savefig('q_align_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("����ͼ�ѱ���: q_align_flow_diagram.png")
    
    # ��������Ҫ��ͼ
    fig2 = create_input_requirements_diagram()
    fig2.savefig('q_align_input_requirements.png', dpi=300, bbox_inches='tight')
    print("����Ҫ��ͼ�ѱ���: q_align_input_requirements.png")
    
    plt.show()

if __name__ == "__main__":
    main()
