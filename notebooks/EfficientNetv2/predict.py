# ============================================================
#  predict.py — Dự đoán bệnh cho 1 ảnh mới
#  Cách dùng: python predict.py duong_dan_anh.jpg
# ============================================================

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import sys
import os

# ============================================================
# CẤU HÌNH
# ============================================================
MODEL_PATH = r"d:\dataset\best_model.pth"
CONFIDENCE_THRES = 0.65   # Dưới ngưỡng này → "Không xác định"

# ── Mô tả ngắn cho từng class (tùy chỉnh theo dataset của bạn) ──
CLASS_INFO = {
    "cafe_Healthy"          : "Cafe - La khoe manh",
    "cafe_Leaf rust"        : "Cafe - Gi sat la",
    "cafe_Miner"            : "Cafe - Sau ve bua",
    "cafe_Phoma"            : "Cafe - Benh Phoma",
    "rice_Healthy Rice Leaf": "Lua - La lua khoe manh",
    "rice_Leaf Blast"       : "Lua - Dao on la",
    "rice_Leaf scald"       : "Lua - Chay la lua",
    "rice_Sheath Blight"    : "Lua - Kho van lua",
    "tieu__Healthy"         : "Tieu - La tieu khoe manh",
    "tieu_Footrot"          : "Tieu - Chet nhanh (Footrot)",
    "tieu_Pollu_Disease"    : "Tieu - Benh tieu dien",
    "tieu_Slow-Decline"     : "Tieu - Vang la (Slow Decline)",
}

if len(sys.argv) < 2:
    print("Cach dung: python predict.py <duong dan anh>")
    print("Vi du   : python predict.py test_leaf.jpg")
    sys.exit(1)

IMG_PATH = sys.argv[1]

if not os.path.exists(IMG_PATH):
    print(f"[!] Khong tim thay anh: {IMG_PATH}")
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"[!] Khong tim thay model: {MODEL_PATH}")
    print("    Hay chay train.py truoc!")
    sys.exit(1)

# ============================================================
# LOAD MODEL
# ============================================================
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["classes"]
num_classes = len(class_names)
model_name  = checkpoint.get("model_name", "mobilevit_s")  # tuong thich model cu

model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

# ============================================================
# TRANSFORM + PREDICT
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

img    = Image.open(IMG_PATH).convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(tensor)
    probs   = F.softmax(outputs, dim=1)[0]

top3_probs, top3_idx = torch.topk(probs, k=min(3, num_classes))
top_prob  = top3_probs[0].item()
top_class = class_names[top3_idx[0]]

# ============================================================
# HIEN THI KET QUA
# ============================================================
fname = os.path.basename(IMG_PATH)
w, h  = img.size

SEP = "=" * 56

print()
print(SEP)
print("   KET QUA NHAN DIEN BENH CAY TRONG")
print(SEP)
print(f"  Anh    : {fname}")
print(f"  Kich thuoc : {w} x {h} px")
print(f"  Model  : {model_name}  |  Device: {device}")
print(SEP)

if top_prob < CONFIDENCE_THRES:
    # ── Khong xac dinh ─────────────────────────────────────
    print("  [!] KHONG XAC DINH")
    print("      Anh co the khong phai la cay trong dataset")
    print(f"      Do tin cay cao nhat : {top_prob*100:.1f}%")
    print(f"      (Nguong toi thieu   : {CONFIDENCE_THRES*100:.0f}%)")
    print(SEP)
    print("  Top du doan gan nhat (chi tham khao):")
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_idx)):
        cname = class_names[idx]
        info  = CLASS_INFO.get(cname, cname)
        print(f"    {i+1}. {info:<38} {prob*100:5.1f}%")
else:
    # ── Ket qua hop le ─────────────────────────────────────
    info = CLASS_INFO.get(top_class, top_class)

    # Thanh progress
    bar_len = 30
    filled  = int(top_prob * bar_len)
    bar     = "#" * filled + "-" * (bar_len - filled)

    print(f"  [OK] KET QUA CHINH")
    print(f"       {info}")
    print(f"       Do tin cay : {top_prob*100:5.1f}%  [{bar}]")
    print(SEP)
    print("  Top 3 du doan:")
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_idx)):
        cname  = class_names[idx]
        info_i = CLASS_INFO.get(cname, cname)
        marker = " <-- KET QUA CHINH" if i == 0 else ""
        print(f"    {i+1}. {info_i:<38} {prob*100:5.1f}%{marker}")

    # Canh bao neu top-1 va top-2 qua gan nhau
    if len(top3_probs) >= 2:
        gap = (top3_probs[0] - top3_probs[1]).item()
        if gap < 0.15:
            print(SEP)
            print(f"  [!] Hai benh hang dau chenh nhau {gap*100:.1f}% -- nen kiem tra them")

print(SEP)
print()