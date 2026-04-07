import os, time, math, torch, torch.nn as nn, timm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torch.amp import GradScaler, autocast
from collections import Counter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ============================================================
# CẤU HÌNH
# ============================================================
DATA_DIR    = r"d:\dataset"          # train/ val/ test/ theo thư mục class
NUM_EPOCHS  = 50
LR          = 2e-4
SAVE_PATH   = r"d:\best_mobilevit.pth"
LOG_PATH    = r"d:\train_log.txt"
PLOT_PATH   = r"d:\train_plot.png"
CM_PATH     = r"d:\confusion_matrix.png"

# ── Auto-detect GPU ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 16:  BATCH_SIZE = 64
    elif vram_gb >= 8: BATCH_SIZE = 32
    else:              BATCH_SIZE = 16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# ============================================================
# AUGMENTATION
# ── MobileViT dùng ảnh 256x256 (khác EfficientNet dùng 224)
# ============================================================
IMG_SIZE = 256

train_tf = transforms.Compose([
    transforms.RandomChoice([
        transforms.Resize((288, 288)),
        transforms.Resize((320, 320)),
        transforms.Resize((352, 352)),
    ]),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                            scale=(0.8, 1.2), shear=8),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    # MobileViT dùng mean/std của ImageNet (khác code cũ dùng 0.5)
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ============================================================
# EVALUATE — trả về loss, accuracy
# ============================================================
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                out  = model(imgs)
                loss = criterion(out, lbls)
            total_loss    += loss.item() * imgs.size(0)
            total_correct += (out.argmax(1) == lbls).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_correct / n * 100

# ============================================================
# CONFUSION MATRIX — 16 lớp bệnh
# ============================================================
def save_confusion_matrix(model, loader, class_names, path):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig_size = max(12, len(class_names))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_title("Confusion Matrix (normalized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix lưu tại: {path}")

    # In classification report ra console
    print("\n  📋 Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=3))

# ============================================================
# VẼ BIỂU ĐỒ LOSS & ACCURACY
# ============================================================
def save_plot(log_lines, path):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    for line in log_lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        e, tl, ta, vl, va = parts
        epochs.append(int(e))
        train_loss.append(float(tl)); train_acc.append(float(ta))
        val_loss.append(float(vl));   val_acc.append(float(va))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MobileViT Training Progress — Plant Disease Classification",
                 fontsize=13, fontweight="bold")

    axes[0].plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, val_loss,   "r-o", markersize=3, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, "b-o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, val_acc,   "r-o", markersize=3, label="Val Acc")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim([max(0, min(val_acc) - 5), 101])
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    best_epoch = val_acc.index(max(val_acc))
    axes[1].annotate(f"Best: {max(val_acc):.2f}%",
                     xy=(epochs[best_epoch], max(val_acc)),
                     xytext=(epochs[best_epoch], max(val_acc) - 5),
                     arrowprops=dict(arrowstyle="->", color="green"),
                     fontsize=9, color="green")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Biểu đồ lưu tại: {path}")

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  GIAI ĐOẠN 2: MobileViT — Phân loại bệnh cây trồng")
    print("=" * 60)
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {gpu_name}")
        print(f"  VRAM   : {vram_gb:.1f} GB")
    print(f"  Batch  : {BATCH_SIZE}  |  ImgSize : {IMG_SIZE}x{IMG_SIZE}")
    print("=" * 60)

    # ── Load dataset ────────────────────────────────────────
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_data   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)
    test_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_tf)

    NUM_CLASSES  = len(train_data.classes)
    class_names  = train_data.classes

    print(f"\n  Classes ({NUM_CLASSES}):")
    for i, c in enumerate(class_names):
        print(f"    [{i:02d}] {c}")

    # ── Class imbalance check ───────────────────────────────
    label_counts = Counter(train_data.targets)
    print("\n  Số ảnh mỗi class (train):")
    for idx, cnt in sorted(label_counts.items()):
        bar = "█" * (cnt // 50)
        print(f"    {class_names[idx]:<35}: {cnt:>5}  {bar}")

    # ── Weighted Sampler để cân bằng class ─────────────────
    weights = [1.0 / label_counts[t] for t in train_data.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    print(f"\n  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
    print("=" * 60)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── MobileViT-Small (model chính cho báo cáo) ───────────
    # Thử các tên model theo thứ tự ưu tiên
    MODEL_CANDIDATES = [
        "mobilevit_s",          # MobileViT-Small — model chính
        "mobilevit_xs",         # MobileViT-XS (nhỏ hơn, nếu VRAM thấp)
        "mobilevitv2_100",      # MobileViTv2 (cải tiến)
        "mobilevitv2_075",      # MobileViTv2 nhỏ hơn
    ]

    model = None
    model_name_used = None
    for _name in MODEL_CANDIDATES:
        try:
            model = timm.create_model(
                _name,
                pretrained=True,
                num_classes=NUM_CLASSES,
                drop_rate=0.2,          # dropout nhẹ hơn EfficientNet vì ViT đã có regularization
            ).to(device)
            model_name_used = _name
            print(f"\n  ✅ Model   : {_name}")
            break
        except Exception as e:
            print(f"  ⚠  {_name} không load được: {e}")
            continue

    if model is None:
        raise RuntimeError("Không load được model nào! Kiểm tra lại timm version.")

    # In tóm tắt số params
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params : {total_params/1e6:.2f}M total | {trainable_params/1e6:.2f}M trainable")

    # ── Loss & Optimizer ────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # MobileViT thường hội tụ tốt hơn với AdamW + weight decay nhẹ
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # ── Learning Rate Scheduler ─────────────────────────────
    # Warmup 5 epoch → Cosine Annealing (giống báo cáo)
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, NUM_EPOCHS - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler()

    # ── Training loop ───────────────────────────────────────
    best_val_acc = 0.0
    patience     = 12       # early stopping
    no_improve   = 0
    log_lines    = ["Epoch,TrainLoss,TrainAcc,ValLoss,ValAcc"]

    print(f"\n  Bắt đầu training MobileViT ({NUM_EPOCHS} epochs)...\n")

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, total_correct = 0.0, 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]",
                    ncols=90)

        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                out  = model(imgs)
                loss = criterion(out, lbls)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += (out.argmax(1) == lbls).sum().item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}"
            })

        scheduler.step()
        train_loss = total_loss / len(train_data)
        train_acc  = total_correct / len(train_data) * 100
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:02d} | "
              f"Train {train_loss:.4f} / {train_acc:.1f}% | "
              f"Val {val_loss:.4f} / {val_acc:.1f}% | "
              f"{elapsed:.0f}s")

        log_lines.append(
            f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f}"
        )
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

        # ── Lưu model tốt nhất ─────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save({
                "epoch":        epoch + 1,
                "model_state":  model.state_dict(),
                "val_acc":      val_acc,
                "classes":      class_names,
                "model_name":   model_name_used,
                "img_size":     IMG_SIZE,
                "num_classes":  NUM_CLASSES,
            }, SAVE_PATH)
            print(f"  ✓ Saved best model → val_acc = {val_acc:.2f}%\n")
        else:
            no_improve += 1
            print(f"  · No improve {no_improve}/{patience}\n")
            if no_improve >= patience:
                print(f"  ⏹  Early stopping tại epoch {epoch+1}")
                break

    # ── Vẽ biểu đồ ─────────────────────────────────────────
    save_plot(log_lines, PLOT_PATH)

    # ── Test set evaluation ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  ĐÁNH GIÁ TRÊN TEST SET")
    print("=" * 60)

    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"  Test Accuracy  : {test_acc:.2f}%")
    print(f"  Best Val Acc   : {best_val_acc:.2f}%")
    print(f"  Model lưu tại  : {SAVE_PATH}")

    # ── Confusion Matrix (quan trọng cho báo cáo mục 4.3.3) ─
    save_confusion_matrix(model, test_loader, class_names, CM_PATH)

    print("=" * 60)
    print("  ✅ Training hoàn tất!")
    print("=" * 60)