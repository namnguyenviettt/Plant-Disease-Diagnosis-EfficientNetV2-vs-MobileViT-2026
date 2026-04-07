from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Generator

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """Bạn là chuyên gia nông nghiệp AI với hơn 20 năm kinh nghiệm tại Việt Nam.

PHONG CÁCH TRẢ LỜI:
- Ngắn gọn, súc tích, dễ hiểu cho nông dân
- Dùng emoji phù hợp để dễ đọc (🌿 🔬 💊 ⚠️ ✅)
- Trả lời bằng tiếng Việt
- Ưu tiên giải pháp thực tế, có thể áp dụng ngay
- Nếu không chắc chắn, hãy nói rõ và khuyên tham khảo chuyên gia địa phương

PHẠM VI CHUYÊN MÔN:
- Bệnh cây trồng (lúa, rau củ, cây ăn trái, cây công nghiệp)
- Sâu bệnh và côn trùng gây hại
- Phân bón và dinh dưỡng cây
- Kỹ thuật canh tác
- Thời tiết và mùa vụ phù hợp Việt Nam

Nếu câu hỏi ngoài phạm vi nông nghiệp, hãy lịch sự từ chối và hướng về chủ đề cây trồng."""


def ask_chatgpt(messages: list[dict]) -> str:
    """Gọi ChatGPT với toàn bộ lịch sử hội thoại."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        temperature=0.7,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def ask_chatgpt_stream(messages: list[dict]) -> Generator[str, None, None]:
    """Streaming version — yield từng chunk text."""
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        temperature=0.7,
        max_tokens=1000,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def build_disease_analysis_prompt(disease: str) -> str:
    """Tạo prompt phân tích bệnh chi tiết và có cấu trúc."""
    return f"""Cây vừa được AI chẩn đoán mắc bệnh: **{disease}**

Hãy cung cấp thông tin đầy đủ theo cấu trúc sau:

🔬 **Bệnh:** {disease}

🌱 **Nguyên nhân:**
(Giải thích ngắn gọn nguyên nhân gây bệnh)

⚠️ **Dấu hiệu nhận biết:**
(Các triệu chứng điển hình trên lá, thân, quả)

💊 **Cách điều trị:**
(Các bước xử lý cụ thể, tên thuốc nếu có)

🛡️ **Phòng tránh:**
(Biện pháp ngăn ngừa tái phát)

⏰ **Mức độ khẩn cấp:** (Thấp / Trung bình / Cao)"""