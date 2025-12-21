import os
import argparse
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import get_cosine_schedule_with_warmup

# 사용자 정의 모듈 임포트 (경로에 맞게 수정)
from src.utils import set_seed
from src.models.sgch import SGCH_Net
from src.dataloader import get_loader
from src.trainer import SGTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# def plot_attention(attn_map, epoch, save_dir):
#     attn_dir = os.path.join(save_dir, "attn_maps")
#     os.makedirs(attn_dir, exist_ok=True)

#     attn_maps = [
#         ("Hybrid", attn_map[0]),
#         ("Skeleton", attn_map[1]),
#         ("Self", attn_map[2]),
#         ("RGB", attn_map[3]),
#     ]

#     plt.figure(figsize=(24, 5))

#     for i, (name, attn) in enumerate(attn_maps):
#         plt.subplot(1, 4, i + 1)
#         a = attn[0]  # (nhead, Q, K) 또는 (Q, K)

#         if a.dim() == 3: m = a.mean(dim=0) # head 처리
#         elif a.dim() == 2: m = a           # head 처리
#         else: raise ValueError(f"Unexpected attn shape: {a.shape}")
#         m = m.detach().cpu().numpy()
#         if m.ndim == 3: m = m.mean(axis=0) # numpy 기준 방어

#         sns.heatmap(m, cmap="viridis")
#         plt.title(name)
#         plt.xlabel("Key")
#         plt.ylabel("Query")

#     plt.savefig(
#         os.path.join(attn_dir, f"attn_triplet_{epoch}.jpg"),
#         dpi=300,
#         bbox_inches="tight"
#     )
#     plt.close()
def plot_attention(analysis_data, epoch, save_dir):
    attn_dir = os.path.join(save_dir, "analysis_plots")
    os.makedirs(attn_dir, exist_ok=True)

    # 1. 데이터 추출 (분석 데이터 구조에 맞게 언패킹)
    skel_hybrid = analysis_data['skel_attn']             # (B*T, nhead, 21, 21)
    f_skel, f_self, f_rgb = analysis_data['fusion_attns'] # 각각 (B*T, nhead, Q, K)
    temp_scores = analysis_data['temporal_scores']       # (B, 1, T) 또는 (B, T)

    # 시각화할 맵 리스트 (이름, 데이터)
    # Temporal을 제외한 4개의 공간 어텐션
    spatial_maps = [
        ("Hybrid (Graph)", skel_hybrid),
        ("Fusion (Skel Ref)", f_skel),
        ("Fusion (Self)", f_self),
        ("Fusion (RGB Ref)", f_rgb),
    ]

    # 1행 5열 구성 (공간 4개 + 시간 점수 1개)
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # [Part 1] 공간 어텐션 히트맵 그리기
    for i, (name, attn) in enumerate(spatial_maps):
        ax = axes[i]
        # 첫 번째 배치/프레임 선택 및 헤드 평균
        # attn: (B*T, nhead, Q, K) -> a: (nhead, Q, K)
        a = attn[0] 
        
        if a.dim() == 3: # (nhead, Q, K)
            m = a.mean(dim=0)
        else:
            m = a
            
        m = m.detach().cpu().numpy()
        sns.heatmap(m, cmap="viridis", ax=ax)
        ax.set_title(f"{name}\n(Epoch {epoch})")
        ax.set_xlabel("Key Index")
        ax.set_ylabel("Query (Joints)")

    # [Part 2] Temporal Score 시각화 (마지막 5번째 칸)
    ax_temp = axes[4]
    # temp_scores: (B, 1, T) -> s: (T,)
    s = temp_scores[0].detach().cpu().numpy().flatten()
    
    # 시간 축(프레임 번호) 생성
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

def main(args):
    config = load_config(args.config)
    exp_name = config['exp_name']

    save_dir = os.path.join("results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config_backup.yaml"), "w") as f:
        yaml.dump(config, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config['seed'])

    df = pd.read_csv(config['data']['csv_path'])
    num_classes = len(df['label'].unique())
    # num_frames = int(df["frames"].iloc[0])

    print("🤖 Initializing Model...")
    model = SGCH_Net(
        num_classes=num_classes,
    ).to(device)

    print("📚 Loading Data...")
    train_loader, valid_loader = get_loader(
        config['data']['data_dir'],
        batch_size=config['train']['batch_size']
    )

    print("🖥️ Loading Trainer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['train']['lr']),
        weight_decay=float(config['train'].get('weight_decay', 0.0))
    )
    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * config['train']['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config['train']['warmup_ratio']),  # e.g. 0.05
        num_training_steps=total_steps
    )
    trainer = SGTrainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    best_f1 = 0.0
    history = [] # 로그 저장용 리스트
    for epoch in range(config['train']['epochs']):
        train_acc, train_loss = trainer.train_epoch(epoch)
        val_acc, val_loss, f1_macro, cm, attn_maps = trainer.validation(valid_loader)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | f1: {f1_macro:.2f}")
        
        plot_attention(attn_maps, epoch, save_dir)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": val_loss,
            "val_mrr": val_acc,
            "f1_macro": f1_macro
        })
        
        if f1_macro > best_f1:
            print(f"✅ Best Model Updated! ({best_f1:.4f} -> {f1_macro:.4f})")
            best_f1 = f1_macro
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_sgch.pt"))
            
        pd.DataFrame(history).to_csv(os.path.join(save_dir, "logs.csv"), index=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Validation)")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.jpg"), dpi=300, bbox_inches="tight")
    plt.close() # 메모리 정리

    print("✨ Experiment Finished!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 실행할 때 --config 옵션으로 yaml 파일 경로를 받음
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    main(args)