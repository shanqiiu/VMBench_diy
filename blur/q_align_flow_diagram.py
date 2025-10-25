# -*- coding: utf-8 -*-
"""
Q-Align算法流程图生成器
用于可视化Q-Align算法的工作流程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_q_align_flow_diagram():
    """创建Q-Align算法流程图"""
    
    # 设置图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E3F2FD',
        'preprocess': '#F3E5F5',
        'vision': '#E8F5E8',
        'language': '#FFF3E0',
        'output': '#FFEBEE',
        'arrow': '#666666'
    }
    
    # 1. 输入阶段
    input_box = FancyBboxPatch(
        (0.5, 10.5), 2, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['input'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(1.5, 11, '视频输入\n(Video Input)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 2. 视频预处理
    preprocess_box = FancyBboxPatch(
        (3.5, 10.5), 2, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['preprocess'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(preprocess_box)
    ax.text(4.5, 11, '滑动窗口\n(Sliding Window)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 3. 图像预处理
    img_preprocess_box = FancyBboxPatch(
        (6.5, 10.5), 2, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['preprocess'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(img_preprocess_box)
    ax.text(7.5, 11, '图像预处理\n(Image Preprocess)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 4. 视觉编码器
    vision_box = FancyBboxPatch(
        (1, 8.5), 2.5, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['vision'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(vision_box)
    ax.text(2.25, 9, 'CLIP视觉编码器\n(CLIP Vision Encoder)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 5. 视觉抽象器
    visual_abstractor_box = FancyBboxPatch(
        (4, 8.5), 2.5, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['vision'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(visual_abstractor_box)
    ax.text(5.25, 9, '视觉抽象器\n(Visual Abstractor)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 6. 语言模型
    language_box = FancyBboxPatch(
        (7, 8.5), 2.5, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['language'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(language_box)
    ax.text(8.25, 9, 'LLaMA语言模型\n(LLaMA Language Model)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 7. 偏好学习头
    preference_box = FancyBboxPatch(
        (2, 6.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(preference_box)
    ax.text(3.5, 7, '偏好学习头\n(Preference Learning Head)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 8. 质量分数
    quality_box = FancyBboxPatch(
        (6, 6.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(quality_box)
    ax.text(7.5, 7, '质量分数\n(Quality Score)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 9. 模糊检测
    blur_detection_box = FancyBboxPatch(
        (1, 4.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(blur_detection_box)
    ax.text(2.5, 5, '模糊检测\n(Blur Detection)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 10. 输出结果
    output_box = FancyBboxPatch(
        (5, 4.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(6.5, 5, '检测结果\n(Detection Results)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 添加箭头连接
    arrows = [
        # 输入到预处理
        ((2.5, 10.5), (3.5, 10.5)),
        # 预处理到图像预处理
        ((5.5, 10.5), (6.5, 10.5)),
        # 图像预处理到视觉编码器
        ((7.5, 10), (2.25, 9.5)),
        # 视觉编码器到视觉抽象器
        ((3.5, 9), (4, 9)),
        # 视觉抽象器到语言模型
        ((6.5, 9), (7, 9)),
        # 语言模型到偏好学习头
        ((8.25, 8), (3.5, 7.5)),
        # 偏好学习头到质量分数
        ((5, 7), (6, 7)),
        # 质量分数到模糊检测
        ((7.5, 6), (2.5, 5.5)),
        # 模糊检测到输出结果
        ((4, 5), (5, 5))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
        ax.add_patch(arrow)
    
    # 添加详细说明
    details = [
        (0.5, 9.5, "? 视频帧序列\n? PIL.Image格式\n? RGB颜色空间"),
        (3.5, 9.5, "? 滑动窗口\n? 帧填充\n? 时间对齐"),
        (6.5, 9.5, "? 正方形扩展\n? 尺寸标准化\n? 颜色归一化"),
        (1, 8, "? CLIP ViT\n? 视觉特征提取\n? 多尺度编码"),
        (4, 8, "? 特征抽象\n? 语言对齐\n? 表示学习"),
        (7, 8, "? LLaMA-2\n? 文本理解\n? 多模态融合"),
        (2, 6, "? 质量等级\n? 偏好学习\n? 权重计算"),
        (6, 6, "? 0-1分数\n? 概率分布\n? 置信度"),
        (1, 3.5, "? 分数差异\n? 阈值检测\n? 异常识别"),
        (5, 3.5, "? 模糊帧位置\n? 严重程度\n? 改进建议")
    ]
    
    for x, y, text in details:
        ax.text(x, y, text, fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 添加标题
    ax.text(5, 11.8, 'Q-Align算法流程图', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['input'], label='输入阶段'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['preprocess'], label='预处理阶段'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['vision'], label='视觉处理'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['language'], label='语言处理'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['output'], label='输出阶段')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def create_input_requirements_diagram():
    """创建输入要求图表"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(5, 7.5, 'Q-Align输入要求详解', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # 输入格式
    input_format_box = FancyBboxPatch(
        (0.5, 6), 4, 1,
        boxstyle="round,pad=0.1",
        facecolor='#E3F2FD',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_format_box)
    ax.text(2.5, 6.5, '输入格式 (Input Format)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # 视频帧要求
    video_requirements = """
视频帧要求 (Video Frame Requirements):
? 类型: PIL.Image
? 颜色空间: RGB
? 尺寸: 任意（自动调整）
? 数量: 根据window_size确定
? 格式: List[List[PIL.Image]]
    """
    
    ax.text(0.5, 5, video_requirements, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.8))
    
    # 评估提示
    prompt_requirements = """
评估提示 (Evaluation Prompt):
? 默认: "USER: How would you rate the quality of this video?\\n<|image|>\\nASSISTANT: The quality of the video is"
? 结构: USER + <|image|> + ASSISTANT
? 占位符: <|image|> 用于图像插入
    """
    
    ax.text(5.5, 5, prompt_requirements, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.8))
    
    # 滑动窗口
    window_requirements = """
滑动窗口 (Sliding Window):
? window_size: 3-5帧
? 填充策略: 边界帧重复
? 时间对齐: 中心帧对齐
? 重叠: 相邻窗口重叠
    """
    
    ax.text(0.5, 3, window_requirements, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.8))
    
    # 质量等级
    quality_levels = """
质量等级 (Quality Levels):
? excellent: 1.0 (优秀)
? good: 0.75 (良好)
? fair: 0.5 (一般)
? poor: 0.25 (较差)
? bad: 0.0 (很差)
    """
    
    ax.text(5.5, 3, quality_levels, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.8))
    
    # 代码示例
    code_example = """
代码示例 (Code Example):
```python
# 初始化模型
scorer = QAlignVideoScorer(device="cuda:0")

# 加载视频
video_frames = load_video_sliding_window("video.mp4", window_size=3)

# 评估质量
_, _, scores = scorer(video_frames)
```
    """
    
    ax.text(0.5, 1, code_example, fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F5E8', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("生成Q-Align算法流程图...")
    
    # 创建流程图
    fig1 = create_q_align_flow_diagram()
    fig1.savefig('q_align_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("流程图已保存: q_align_flow_diagram.png")
    
    # 创建输入要求图
    fig2 = create_input_requirements_diagram()
    fig2.savefig('q_align_input_requirements.png', dpi=300, bbox_inches='tight')
    print("输入要求图已保存: q_align_input_requirements.png")
    
    plt.show()

if __name__ == "__main__":
    main()
