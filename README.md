# 🍃 AI Plant Disease Diagnosis: EfficientNetV2 vs MobileViT

> **Đồ án môn học Deep Learning:** Nghiên cứu đối sánh cấu trúc mạng thuần CNN (EfficientNetV2) và mạng lai Transformer (MobileViT) trong nhận diện bệnh lý trên lá cây (Cà phê, Lúa, Hồ tiêu), kết hợp hệ thống tư vấn phác đồ điều trị từ OpenAI.

## 📋 Giới thiệu chung
Dự án này cung cấp một hệ thống chẩn đoán bệnh thực vật hoàn chỉnh từ khâu huấn luyện mô hình (Training) đến khâu triển khai máy chủ (Deployment) qua API. 
* **Mô hình cốt lõi:** Dựa trên kết quả thực nghiệm, MobileViT-S được chọn để triển khai nhờ ưu thế vượt trội về dung lượng siêu nhẹ (chỉ 4.95M tham số) và hiệu năng hội tụ ổn định, cực kỳ phù hợp cho môi trường Edge AI.
* **Hệ thống API:** Được xây dựng bằng kiến trúc Client-Server (Flask/FastAPI). Máy chủ tiếp nhận ảnh, suy luận bằng mô hình Học sâu, sau đó kết nối với OpenAI API để sinh phác đồ điều trị theo thời gian thực trước khi trả kết quả về cho người dùng.

## ✨ Các tính năng nổi bật
- 📷 **Nhận diện khá chính xác:** Phân loại các loại bệnh phổ biến trên cây Cà phê, Lúa và Hồ tiêu.
- ⚡ **Tối ưu tài nguyên:** Suy luận cực nhanh với mô hình nhẹ, không gây quá tải cho bộ nhớ máy chủ.
- 🤖 **Trợ lý ảo Nông nghiệp:** Tự động cung cấp hướng dẫn dùng thuốc và xử lý bệnh từ Mô hình ngôn ngữ lớn (LLM).

.
├── notebooks/              
├── server/                  
├── models/                 
├── README.md               