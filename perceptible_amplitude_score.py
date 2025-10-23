import argparse
import os
import sys
import math
from tqdm import tqdm

import numpy as np
import json
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything", "segment_anything"))
sys.path.append(os.path.join(os.getcwd(), "co-tracker"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)


# Co-Tracker
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None, None, None, None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        print("Error reading frames from the video")
        return None, None, None, None

    # take the first frame as the query image
    frame_rgb = frames[0]
    image_pil = Image.fromarray(frame_rgb)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    
    return image_pil, image, frame_rgb, np.stack(frames)


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def is_mask_suitable_for_tracking(mask, video_width, video_height, grid_size, min_area_ratio=0.001):
    """
    检查掩码是否适合进行跟踪
    
    Parameters:
    mask: torch.Tensor, 掩码
    video_width: int, 视频宽度
    video_height: int, 视频高度
    grid_size: int, 网格大小
    min_area_ratio: float, 最小区域比例
    
    Returns:
    bool: 是否适合跟踪
    """
    mask_area = torch.sum(mask > 0).item()
    total_area = video_width * video_height
    area_ratio = mask_area / total_area
    
    # 检查区域是否足够大
    if area_ratio < min_area_ratio:
        return False
    
    # 检查区域是否足够生成网格点
    # 粗略估计：需要至少 (grid_size/2)^2 个像素
    min_pixels_needed = (grid_size // 2) ** 2
    if mask_area < min_pixels_needed:
        return False
    
    return True


def calculate_motion_degree(keypoints, video_width, video_height):
    """
    Calculate the normalized motion amplitude for each batch sample
    
    Parameters:
    keypoints: torch.Tensor, shape [batch_size, 49, 792, 2]
    video_width: int, width of the video
    video_height: int, height of the video
    
    Returns:
    motion_amplitudes: torch.Tensor, shape [batch_size], containing the normalized motion amplitude for each batch sample
    """

    # Calculate the length of the video diagonal
    diagonal = torch.sqrt(torch.tensor(video_width**2 + video_height**2, dtype=torch.float32))
    
    # Compute the Euclidean distance between adjacent frames
    distances = torch.norm(keypoints[:, 1:] - keypoints[:, :-1], dim=3)  # shape [batch_size, 48, 792]
    
    # Normalize the distances by the diagonal length to eliminate resolution effects
    normalized_distances = distances / diagonal
    
    # Sum the normalized distances to get the total normalized motion distance for each keypoint
    total_normalized_distances = torch.sum(normalized_distances, dim=1)  # shape [batch_size, 792]
    
    # Compute the normalized motion amplitude for each batch sample (mean of total normalized motion distance for all points)
    motion_amplitudes = torch.mean(total_normalized_distances, dim=1)  # shape [batch_size]
    
    return motion_amplitudes


def visualize_detection(image, boxes, labels, output_path):
    """可视化对象检测结果"""
    vis_image = image.copy()
    
    # 检查是否有检测结果
    if len(boxes) == 0:
        # 如果没有检测到对象，添加文本说明
        cv2.putText(vis_image, "No objects detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # 绘制检测到的对象
        for i, (box, label) in enumerate(zip(boxes, labels)):
            # 绘制边界框
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            cv2.putText(vis_image, f"{label}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    return vis_image


def visualize_masks(image, masks, output_path, alpha=0.5):
    """可视化分割掩码"""
    vis_image = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, mask in enumerate(masks):
        # Ensure mask is 2D boolean array
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # Convert to boolean if needed
        if mask.dtype != bool:
            mask = mask.astype(bool)
            
        # Check if mask dimensions match image dimensions
        if mask.shape != image.shape[:2]:
            print(f"Warning: Mask {i} shape {mask.shape} doesn't match image shape {image.shape[:2]}")
            # Resize mask to match image dimensions
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        color = colors[i % len(colors)]
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        vis_image = cv2.addWeighted(vis_image, 1-alpha, colored_mask, alpha, 0)
    
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    return vis_image


def visualize_tracks(video, tracks, visibility, output_path, grid_size=30):
    """可视化运动轨迹"""
    # 检查tracks是否为空或无效
    if tracks is None or tracks.numel() == 0 or tracks.shape[2] == 0:
        print(f"Warning: No valid tracks to visualize, skipping {output_path}")
        return None
    
    try:
        # 禁用自动保存，我们手动保存到指定路径
        visualizer = Visualizer(save_dir=None)
        vis_frames_tensor = visualizer.visualize(video, tracks, visibility, save_video=False)
        
        # 保存可视化视频
        if vis_frames_tensor.numel() > 0:
            # Convert tensor to numpy array and reshape to (T, H, W, C)
            if vis_frames_tensor.shape[0] == 1:
                vis_frames_np = vis_frames_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
            else:
                vis_frames_np = vis_frames_tensor[0].permute(0, 2, 3, 1).cpu().numpy()
            
            height, width = vis_frames_np[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 8.0, (width, height))
            
            for frame in vis_frames_np:
                # Ensure frame is in correct format for cv2.cvtColor
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                # Ensure frame has correct shape (H, W, C)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                else:
                    print(f"Warning: Invalid frame shape {frame.shape}, skipping...")
            out.release()
        else:
            print(f"Warning: No frames generated for visualization, skipping {output_path}")
        
        return vis_frames_tensor
        
    except Exception as e:
        print(f"Error during track visualization: {e}")
        print(f"Skipping visualization for {output_path}")
        return None


def visualize_motion_analysis(image, background_mask, subject_mask, 
                             background_motion, subject_motion, 
                             output_path):
    """可视化运动分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))   
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 背景掩码
    axes[0, 1].imshow(background_mask, cmap='gray')
    axes[0, 1].set_title(f'Background Mask\nMotion: {background_motion:.4f}')
    axes[0, 1].axis('off')
    
    # 主体掩码
    axes[1, 0].imshow(subject_mask, cmap='gray')
    axes[1, 0].set_title(f'Subject Mask\nMotion: {subject_motion:.4f}')
    axes[1, 0].axis('off')
    
    # 运动分析
    motion_data = {
        'Background Motion': background_motion,
        'Subject Motion': subject_motion,
        'Pure Subject Motion': max(0, subject_motion - background_motion)
    }
    
    bars = axes[1, 1].bar(motion_data.keys(), motion_data.values())
    axes[1, 1].set_title('Motion Analysis')
    axes[1, 1].set_ylabel('Motion Amplitude')
    
    # 添加数值标签
    for bar, value in zip(bars, motion_data.values()):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_detailed_motion_scores(meta_info, background_motion, subject_motion, has_subject, video_width=None, video_height=None, status="normal", error_reason=None):
    """保存详细的运动分数"""
    if has_subject:
        pure_subject = max(0, subject_motion - background_motion)
        total_motion = background_motion + subject_motion
        motion_ratio = pure_subject / (background_motion + 1e-8)
        
        motion_score = {
            'background_motion': float(background_motion),
            'subject_motion': float(subject_motion),
            'pure_subject_motion': float(pure_subject),
            'total_motion': float(total_motion),
            'motion_ratio': float(motion_ratio)
        }
    else:
        motion_score = {
            'background_motion': float(background_motion),
            'subject_motion': 0.0,
            'pure_subject_motion': 0.0,
            'total_motion': float(background_motion),
            'motion_ratio': 0.0
        }
    
    # Add resolution information for normalization tracking
    if video_width is not None and video_height is not None:
        motion_score['video_resolution'] = {
            'width': int(video_width),
            'height': int(video_height),
            'diagonal': float(torch.sqrt(torch.tensor(video_width**2 + video_height**2)).item()),
            'normalized_to_1080p': True
        }
    
    # Add status information
    motion_score['status'] = status
    if error_reason:
        motion_score['error_reason'] = error_reason
    
    meta_info['perceptible_amplitude_score'] = motion_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--meta_info_path", type=str, required=True, help="path to meta info json")
    parser.add_argument("--text_prompt", type=str, required=False, help="text prompt", 
            default="person. dog. cat. horse. car. ball. robot. bird. bicycle. motorcycle. surfboard. skateboard. bucket. bat. basketball. " \
              "racket. kitten. puppy. fish. laptop. umbrella. wheelchair. drone. scooter. rollerblades. truck. bus. skier. snowboard. " \
              "sled. kayak. canoe. sailboat. guitar. piano. drum. violin. trumpet. saxophone. clarinet. flute. accordion. telescope. " \
              "microscope. treadmill. rope. ladder. swing. tugboat. train.")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--grid_size", type=int, default=30, help="Regular grid size")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    
    # 新增可视化参数
    parser.add_argument("--save_visualization", action="store_true", help="Save visualization results")
    parser.add_argument("--output_vis_dir", type=str, default="./vis_results", help="Output directory for visualizations")
    parser.add_argument("--vis_detection", action="store_true", help="Visualize object detection")
    parser.add_argument("--vis_masks", action="store_true", help="Visualize segmentation masks")
    parser.add_argument("--vis_tracks", action="store_true", help="Visualize motion tracks")
    parser.add_argument("--vis_analysis", action="store_true", help="Visualize motion analysis")
    
    args = parser.parse_args()

    # model cfg
    config_file = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
    grounded_checkpoint = ".cache/groundingdino_swinb_cogcoor.pth"
    bert_base_uncased_path = ".cache/google-bert/bert-base-uncased"
    sam_version = "vit_h"
    sam_checkpoint = ".cache/sam_vit_h_4b8939.pth"
    cotracker_checkpoint = ".cache/scaled_offline.pth"

    device = args.device

    # 创建可视化输出目录
    if args.save_visualization:
        os.makedirs(args.output_vis_dir, exist_ok=True)
        print(f"Visualization results will be saved to: {args.output_vis_dir}")

    # load model
    grounding_model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    # initialize SAM
    sam_predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    # intialize Co-Tracker
    cotracker_model = CoTrackerPredictor(
        checkpoint=cotracker_checkpoint,
        v2=False,
        offline=True,
        window_len=60,
    ).to(device)

    # load meta info json
    with open(args.meta_info_path, 'r') as f:
        meta_infos = json.load(f)
    
    for meta_info in tqdm(meta_infos, desc="Motion Degree: Grounded SAM Segmentation"):
        image_pil, image, image_array, video = load_video(meta_info['filepath'])

        text_prompt = meta_info['subject_noun'] + '.'

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            grounding_model, image, text_prompt, args.box_threshold, args.text_threshold, device=device
        )

        # 可视化检测结果
        if args.save_visualization and args.vis_detection:
            vis_detection_path = os.path.join(args.output_vis_dir, f"detection_{meta_info['index']}.jpg")
            visualize_detection(image_array, boxes_filt, pred_phrases, vis_detection_path)

        # 初始化masks变量，避免未定义错误
        masks = None
        
        # no detect object
        if boxes_filt.shape[0] == 0:
            print(f"can not detect {text_prompt} in {meta_info['prompt']}")
        else:
            sam_predictor.set_image(image_array)

            # convert boxes into xyxy format
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            # run sam model
            masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
            )

            # 可视化分割掩码
            if args.save_visualization and args.vis_masks:
                vis_masks_path = os.path.join(args.output_vis_dir, f"masks_{meta_info['index']}.jpg")
                # SAM returns masks with shape (batch_size, num_masks, H, W), safely remove batch dimension
                masks_np = masks.cpu().numpy()
                if masks_np.shape[0] == 1:
                    masks_np = masks_np.squeeze(0)  # Remove batch dimension: (num_masks, H, W)
                else:
                    # If batch size > 1, take the first batch
                    masks_np = masks_np[0]
                visualize_masks(image_array, masks_np, vis_masks_path)

        # load the input video frame by frame
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video_width, video_height = video.shape[-1], video.shape[-2]
        video = video.to(device)
   
        if boxes_filt.shape[0] != 0 and masks is not None:
            background_mask = torch.any(~masks, dim=0).to(torch.uint8) * 255
        else:
            background_mask = torch.ones((1, video_height, video_width), dtype=torch.uint8, device=device) * 255

        background_mask = background_mask.unsqueeze(0)
        
        pred_tracks, pred_visibility = cotracker_model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=0,
            backward_tracking=True,
            segm_mask=background_mask
        )
        
        if pred_tracks.shape[2] == 0:
            background_motion_degree = 0.0
        else:
            background_motion_degree = calculate_motion_degree(pred_tracks, video_width, video_height).item()

        if args.save_visualization and args.vis_tracks and pred_tracks.shape[2] > 0:
            vis_bg_tracks_path = os.path.join(args.output_vis_dir, f"bg_tracks_{meta_info['index']}.mp4")
            visualize_tracks(video, pred_tracks, pred_visibility, vis_bg_tracks_path, args.grid_size)

        if boxes_filt.shape[0] != 0 and masks is not None:
            subject_mask = torch.any(masks, dim=0).to(torch.uint8) * 255
            subject_mask = subject_mask.unsqueeze(0)
            
            # 检查掩码是否适合跟踪
            subject_mask_valid = torch.sum(subject_mask > 0).item() > 0
            mask_suitable = is_mask_suitable_for_tracking(subject_mask, video_width, video_height, args.grid_size)
            
            if not subject_mask_valid:
                subject_motion_degree = 0.0
                save_detailed_motion_scores(meta_info, background_motion_degree, subject_motion_degree, False, video_width, video_height, 
                                          status="error", error_reason="subject_mask_empty")
            elif not mask_suitable:
                print(f"Warning: Mask unsuitable for tracking in video {meta_info['index']}")
                mask_area = torch.sum(subject_mask > 0).item()
                mask_ratio = mask_area / (video_width * video_height)
                print(f"  Mask area: {mask_area} pixels ({mask_ratio:.4f} of frame)")
                print(f"  Grid size: {args.grid_size}")
                print(f"  Mask too small for effective tracking")
                subject_motion_degree = 0.0
                save_detailed_motion_scores(meta_info, background_motion_degree, subject_motion_degree, False, video_width, video_height,
                                          status="error", error_reason="mask_too_small")
            else:
                pred_tracks, pred_visibility = cotracker_model(
                    video,
                    grid_size=args.grid_size,
                    grid_query_frame=0,
                    backward_tracking=True,
                    segm_mask=subject_mask
                )
                
                if pred_tracks.shape[2] == 0:
                    print(f"Warning: Empty tracks for video {meta_info['index']} despite suitable mask")
                    mask_area = torch.sum(subject_mask > 0).item()
                    mask_ratio = mask_area / (video_width * video_height)
                    print(f"  Mask area: {mask_area} pixels ({mask_ratio:.4f} of frame)")
                    print(f"  Grid size: {args.grid_size}")
                    print(f"  Video shape: {video.shape}")
                    print(f"  Possible causes:")
                    print(f"    - Insufficient motion in subject region")
                    print(f"    - Video quality issues (blur, low resolution)")
                    print(f"    - Subject too static or moving too fast")
                    print(f"    - Co-Tracker algorithm limitations")
                    subject_motion_degree = 0.0
                    save_detailed_motion_scores(meta_info, background_motion_degree, subject_motion_degree, True, video_width, video_height,
                                              status="error", error_reason="empty_tracks")
                else:
                    subject_motion_degree = calculate_motion_degree(pred_tracks, video_width, video_height).item()
                    save_detailed_motion_scores(meta_info, background_motion_degree, subject_motion_degree, True, video_width, video_height,
                                              status="normal")

                if args.save_visualization and args.vis_tracks and subject_mask_valid and pred_tracks.shape[2] > 0:
                    vis_subject_tracks_path = os.path.join(args.output_vis_dir, f"subject_tracks_{meta_info['index']}.mp4")
                    visualize_tracks(video, pred_tracks, pred_visibility, vis_subject_tracks_path, args.grid_size)

            
                if args.save_visualization and args.vis_analysis and subject_mask_valid:
                    vis_analysis_path = os.path.join(args.output_vis_dir, f"motion_analysis_{meta_info['index']}.png")
                    
                    bg_mask_np = background_mask.cpu().numpy()
                    sub_mask_np = subject_mask.cpu().numpy()
                    
                    while len(bg_mask_np.shape) > 2 and 1 in bg_mask_np.shape:
                        for i, size in enumerate(bg_mask_np.shape):
                            if size == 1:
                                bg_mask_np = np.squeeze(bg_mask_np, axis=i)
                                break
                    
                    while len(sub_mask_np.shape) > 2 and 1 in sub_mask_np.shape:
                        for i, size in enumerate(sub_mask_np.shape):
                            if size == 1:
                                sub_mask_np = np.squeeze(sub_mask_np, axis=i)
                                break
                    
                    if len(bg_mask_np.shape) > 2:
                        bg_mask_np = bg_mask_np.reshape(bg_mask_np.shape[-2], bg_mask_np.shape[-1])
                    
                    if len(sub_mask_np.shape) > 2:
                        sub_mask_np = sub_mask_np.reshape(sub_mask_np.shape[-2], sub_mask_np.shape[-1])
                    
                    visualize_motion_analysis(image_array, 
                                            bg_mask_np,
                                            sub_mask_np,
                                            background_motion_degree, 
                                            subject_motion_degree,
                                            vis_analysis_path)
        else:
            save_detailed_motion_scores(meta_info, background_motion_degree, 0, False, video_width, video_height,
                                      status="error", error_reason="no_subject_detected")

        with open(args.meta_info_path, 'w') as f:
            json.dump(meta_infos, f, indent=4)

    print("PAS evaluation completed!")
    if args.save_visualization:
        print(f"Visualization results saved to: {args.output_vis_dir}")