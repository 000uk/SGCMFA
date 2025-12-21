from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, confusion_matrix

class SGTrainer:
    def __init__(self, model, train_loader, criterion, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, epoch):     
        self.model.train()
        train_loss = 0
        total = 0
        correct = 0

        for batch in tqdm(self.train_loader, desc=f"[Epoch: {epoch + 1}]"):
            x_rgb = batch.get("rgb")
            x_skel = batch.get("skel")
            targets = batch["label"]

            if x_rgb is not None:
                x_rgb = x_rgb.to(self.device)
            if x_skel is not None:
                x_skel = x_skel.to(self.device)

            targets = targets.to(self.device)
        
            self.optimizer.zero_grad()
            outputs, _ = self.model(x_rgb, x_skel)

            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_loss /= len(self.train_loader)

        return train_acc, train_loss
    
    def validation(self, valid_loader):
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in valid_loader:
                x_rgb = batch.get("rgb")
                x_skel = batch.get("skel")
                targets = batch["label"]

                if x_rgb is not None:
                    x_rgb = x_rgb.to(self.device)
                if x_skel is not None:
                    x_skel = x_skel.to(self.device)

                targets = targets.to(self.device)
                
                # outputs = self.model(x_rgb, x_skel)
                outputs, attn_maps = self.model(x_rgb, x_skel)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        val_acc = 100. * val_correct / val_total
        val_loss /= len(valid_loader)

        f1_macro = f1_score(all_labels, all_preds, average="macro")
        cm = confusion_matrix(all_labels, all_preds)

        return val_acc, val_loss, f1_macro, cm, attn_maps