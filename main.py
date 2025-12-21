import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import get_cosine_schedule_with_warmup

from src.utils import set_seed, plot_attention, plot_confusion_matrix
from src.models.sgch import SGCH_Net
from src.dataloader import get_loader
from src.trainer import SGTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
        
        if f1_macro > best_f1:
            print(f"✅ Best Model Updated! ({best_f1:.4f} -> {f1_macro:.4f})")
            best_f1 = f1_macro
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_sgch.pt"))

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": val_loss,
            "val_mrr": val_acc,
            "f1_macro": f1_macro
        })
            
        pd.DataFrame(history).to_csv(os.path.join(save_dir, "logs.csv"), index=False)
        plot_attention(attn_maps, epoch, save_dir)
    plot_confusion_matrix(cm, save_dir)

    print("✨ Experiment Finished!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 실행할 때 --config 옵션으로 yaml 파일 경로를 받음
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    main(args)