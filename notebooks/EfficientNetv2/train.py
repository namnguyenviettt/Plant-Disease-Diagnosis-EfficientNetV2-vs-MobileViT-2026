import os, time, torch, torch.nn as nn, timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter # [MỚI] Thêm thư viện để đếm số lượng class

# Tắt giới hạn kích thước ảnh của PIL (fix DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# ============================================================
# CẤU HÌNH
# ============================================================
DATA_DIR    = r"d:\dataset"
NUM_EPOCHS  = 40
BATCH_SIZE  = 32          # Tăng từ 16 → 32: nhanh hơn, ổn định hơn
LR          = 3e-4
NUM_WORKERS = 0  # Windows: dùng 0 để tránh multiprocessing crash           # Tăng worker để load data nhanh hơn
SAVE_PATH   = r"d:\dataset\best_model.pth"
LOG_PATH    = r"d:\dataset\train_log.txt"
PLOT_PATH   = r"d:\dataset\train_plot.png"

# ── Auto-detect GPU & tối ưu settings ───────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Tự động điều chỉnh batch size theo VRAM
    if vram_gb >= 16:
        BATCH_SIZE = 64
    elif vram_gb >= 8:
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 16
    # Tăng tốc với TF32 (Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

# ============================================================
# DATA AUGMENTATION — mạnh hơn để tăng accuracy
# ============================================================
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),                          
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    # [MỚI] Chuẩn hóa theo chuẩn ImageNet cho model Pretrained
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  
])

val_tf = transforms.Compose([
    # [MỚI] Sửa lại cách Resize và Crop để giữ nguyên tỷ lệ ảnh
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # [MỚI] Chuẩn hóa theo chuẩn ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# MIXUP AUGMENTATION — trộn 2 ảnh để model tổng quát hơn
# ============================================================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# EVALUATE
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
# VẼ BIỂU ĐỒ
# ============================================================
def save_plot(log_lines, path):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    for line in log_lines[1:]:
        e, tl, ta, vl, va = line.split(",")
        epochs.append(int(e))
        train_loss.append(float(tl)); train_acc.append(float(ta))
        val_loss.append(float(vl));   val_acc.append(float(va))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, val_loss,   "r-o", markersize=3, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, "b-o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, val_acc,   "r-o", markersize=3, label="Val Acc")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim([max(0, min(val_acc) - 5), 101])
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Đánh dấu best val acc
    best_epoch = val_acc.index(max(val_acc))
    axes[1].annotate(f"Best: {max(val_acc):.2f}%",
                     xy=(epochs[best_epoch], max(val_acc)),
                     xytext=(epochs[best_epoch], max(val_acc) - 3),
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
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {gpu_name}")
        print(f"  VRAM   : {vram_gb:.1f} GB")
    print(f"  Batch  : {BATCH_SIZE} (auto-tuned)")
    print("=" * 60)

    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_data   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)
    test_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_tf)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)

    NUM_CLASSES = len(train_data.classes)
    print(f"\n  Classes ({NUM_CLASSES}): {train_data.classes}")
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
    print("=" * 60)

    # ── Model: EfficientNetV2-S — nhanh hơn MobileViT, accuracy cao hơn ──
    for _name in ["tf_efficientnetv2_s", "tf_efficientnetv2_s_in21ft1k", "efficientnetv2_s", "mobilevit_s"]:
        try:
            model = timm.create_model(_name, pretrained=True, num_classes=NUM_CLASSES).to(device)
            model_name_used = _name
            print(f"  Model  : {_name}")
            break
        except Exception:
            continue
            
    # [MỚI] Tính toán Class Weights để xử lý mất cân bằng dữ liệu
    print("  Đang tính toán trọng số cho các class (Class Weights)...")
    class_counts = [Counter(train_data.targets)[i] for i in range(NUM_CLASSES)]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES # Chuẩn hóa trọng số
    
    # [MỚI] Đưa trọng số vào hàm Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    best_val_acc  = 0.0
    patience      = 12    # Early stopping
    no_improve    = 0
    log_lines     = ["Epoch,TrainLoss,TrainAcc,ValLoss,ValAcc"]
    print("\nBắt đầu training...\n")

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, total_correct = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]", ncols=85)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)

            # Áp dụng Mixup
            imgs, y_a, y_b, lam = mixup_data(imgs, lbls, alpha=0.2)

            optimizer.zero_grad()
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                out  = model(imgs)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += (out.argmax(1) == lbls).sum().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()

        # Train loss từ vòng lặp (có Mixup)
        train_loss = total_loss / len(train_data)
        # Tính Train Acc thực sự — eval lại trên train set không qua Mixup
        _, train_acc = evaluate(model, train_loader, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:02d} | Train {train_loss:.4f} {train_acc:.1f}% | Val {val_loss:.4f} {val_acc:.1f}% | {elapsed:.0f}s")
        log_lines.append(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f}")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "classes": train_data.classes,
                "model_name": model_name_used
            }, SAVE_PATH)
            print(f"  ✓ Saved best model → {val_acc:.2f}%\n")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  ⏹  Early stopping tại epoch {epoch+1} (không cải thiện sau {patience} epoch)")
                break

    # ── Vẽ biểu đồ ────────────────────────────────────────────
    save_plot(log_lines, PLOT_PATH)

    # ── Test cuối cùng ────────────────────────────────────────
    print("=" * 60)
    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"  Test Accuracy : {test_acc:.2f}%")
    print(f"  Best Val Acc  : {best_val_acc:.2f}%")
    print(f"  Model lưu tại : {SAVE_PATH}")
    print(f"  Log lưu tại   : {LOG_PATH}")
    print(f"  Plot lưu tại  : {PLOT_PATH}")
    print("=" * 60)