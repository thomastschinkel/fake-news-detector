import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model import FakeNewsConfig, FakeNewsDetector
from tqdm.auto import tqdm
from transformers import RobertaConfig, RobertaTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MAX_LEN       = 512
BATCH_SIZE    = 24
EPOCHS        = 20
LR            = 2e-5
PATIENCE      = 3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS   = 4
USE_AMP       = DEVICE == "cuda"
MODEL_DIR     = Path("models")
MODEL_PATH    = MODEL_DIR / "bert_large.pth"

print(f"Using device: {DEVICE}")

df = pd.read_csv("data/news.csv")[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)
print(f"Dataset size: {len(df):,} articles")

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

class NewsDataset(Dataset):
    def __init__(self, df):
        print("Tokenizing...")
        encodings = tokenizer(
            df["text"].tolist(),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels         = torch.tensor(df["label"].tolist(), dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids":      self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "label":          self.labels[i]
        }

train_dataset = NewsDataset(train_df)
val_dataset   = NewsDataset(val_df)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE * 2,
    num_workers=NUM_WORKERS, pin_memory=True
)

base_config = RobertaConfig.from_pretrained("roberta-large")
model_config = FakeNewsConfig(**base_config.to_dict(), num_labels=2)
model   = FakeNewsDetector(model_config).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

optimizer = AdamW([
    {"params": model.roberta.parameters(),       "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 5e-5}
], weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

best_val_f1    = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    for batch in progress:
        ids    = batch["input_ids"].to(DEVICE, non_blocking=True)
        mask   = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(ids, mask)
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False):
            ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                preds = model(ids, mask).argmax(dim=1).cpu().tolist()
            all_preds  += preds
            all_labels += batch["label"].tolist()

    report = classification_report(
        all_labels, all_preds,
        target_names=["Real", "Fake"],
        output_dict=True
    )
    val_f1  = report["macro avg"]["f1-score"]
    val_acc = report["accuracy"]

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Acc: {val_acc:.2%} | F1: {val_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        tokenizer.save_pretrained("tokenizer/")
        print(f"  New best model saved! (F1: {best_val_f1:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve}/{PATIENCE} epochs")

    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch+1}. "
              f"Best F1: {best_val_f1:.4f}")
        break

print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
print(f"Best model saved to: {MODEL_PATH}")
