import os

# Đường dẫn đến thư mục dataset của bạn
data_dir = r"d:\dataset"

# Hàm đếm ảnh
def count_images_in_dir(directory):
    total_images = 0
    print(f"\n📁 Đang kiểm tra thư mục: {directory}")
    
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(directory):
        print("  -> Thư mục không tồn tại!")
        return
        
    # Duyệt qua từng thư mục con (từng nhãn/class)
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        
        if os.path.isdir(class_path):
            # Lọc và đếm các file có đuôi mở rộng là ảnh
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            count = len(images)
            total_images += count
            print(f"  ├── Class '{class_name}': {count} ảnh")
            
    print(f"  └── TỔNG CỘNG: {total_images} ảnh")

# Chạy đếm cho các tập dữ liệu
for split in ['train', 'val', 'test']:
    split_path = os.path.join(data_dir, split)
    count_images_in_dir(split_path)