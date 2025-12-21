import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_seed(seed: int = 42):
    random.seed(seed)            # 기본 Python random 고정
    np.random.seed(seed)         # NumPy 랜덤 고정
    torch.manual_seed(seed)      # CPU 연산 랜덤 고정
    torch.cuda.manual_seed(seed) # GPU 모든 디바이스 랜덤 고정
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU일 때

    # 연산 재현성
    torch.backends.cudnn.deterministic = True  # cuDNN 연산을 determinisitc으로 강제
    torch.backends.cudnn.benchmark = False     # CUDA 성능 자동 튜닝 기능 끔 → 완전 재현 가능

def sample_clip(num_frames_total, clip_len=30, min_stride=2, max_stride=6):
    if num_frames_total < clip_len:
        return list(range(num_frames_total)) + \
               [num_frames_total - 1] * (clip_len - num_frames_total)

    stride = random.randint(min_stride, max_stride)
    max_start = max(0, num_frames_total - clip_len * stride)
    start = random.randint(0, max_start) if max_start > 0 else 0

    idxs = [start + i * stride for i in range(clip_len)]
    idxs = [min(i, num_frames_total - 1) for i in idxs]

    return idxs

def plot_confusion_matrix(cm, save_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Validation)")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.jpg"), dpi=300, bbox_inches="tight")
    plt.close() # 메모리 정리

def plot_attention(analysis_data, epoch, save_dir):
    attn_dir = os.path.join(save_dir, "analysis_plots")
    os.makedirs(attn_dir, exist_ok=True)

    skel_hybrid = analysis_data['skel_attn']
    f_skel, f_self, f_rgb = analysis_data['fusion_attns']
    temp_scores = analysis_data['temporal_scores']

    spatial_maps = [
        ("Hybrid (Graph)", skel_hybrid),
        ("Fusion (Skel Ref)", f_skel),
        ("Fusion (Self)", f_self),
        ("Fusion (RGB Ref)", f_rgb),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    for i, (name, attn) in enumerate(spatial_maps):
        ax = axes[i]
        a = attn[0] 
        if a.dim() == 3: m = a.mean(dim=0)
        else: m = a
            
        m = m.detach().cpu().numpy()
        sns.heatmap(m, cmap="viridis", ax=ax)
        ax.set_title(f"{name}\n(Epoch {epoch})")
        ax.set_xlabel("Key Index")
        ax.set_ylabel("Query (Joints)")

    ax_temp = axes[4]
    s = temp_scores[0].detach().cpu().numpy().flatten()
    
    frames = np.arange(len(s))
    ax_temp.bar(frames, s, color='salmon', alpha=0.7)
    ax_temp.plot(frames, s, marker='o', color='red', linewidth=2)
    ax_temp.set_ylim(0, 1.1)
    ax_temp.set_title(f"Temporal Importance Score\n(Frame-wise)")
    ax_temp.set_xlabel("Frame Index")
    ax_temp.set_ylabel("Score (0~1)")
    ax_temp.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(
        os.path.join(attn_dir, f"analysis_epoch_{epoch}.jpg"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()