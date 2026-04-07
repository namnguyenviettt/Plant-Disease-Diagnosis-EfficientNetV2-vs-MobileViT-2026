from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json

from predict import predict_disease
from chatgpt_service import ask_chatgpt, ask_chatgpt_stream, build_disease_analysis_prompt

app = FastAPI(title="Plant Doctor AI", version="2.0")


# ============================================================
# SESSION STORE
# Mỗi conversation_id lưu riêng:
#   - disease: bệnh đã chẩn đoán (nếu có)
#   - history: danh sách {role, content} để GPT nhớ ngữ cảnh
# ============================================================

sessions: dict[str, dict] = {}

MAX_HISTORY = 20  # Giữ tối đa 20 lượt để tránh token quá lớn


def get_session(conversation_id: str) -> dict:
    if conversation_id not in sessions:
        sessions[conversation_id] = {
            "disease": None,
            "history": [],
        }
    return sessions[conversation_id]


def trim_history(history: list) -> list:
    """Giữ tối đa MAX_HISTORY tin nhắn gần nhất."""
    if len(history) > MAX_HISTORY:
        return history[-MAX_HISTORY:]
    return history


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    conversation_id: str
    question: str


class NewConversationResponse(BaseModel):
    conversation_id: str


# ============================================================
# API: Tạo conversation mới
# ============================================================

@app.post("/conversation/new", response_model=NewConversationResponse)
async def new_conversation():
    """Tạo session mới, trả về conversation_id."""
    conv_id = str(uuid.uuid4())
    sessions[conv_id] = {"disease": None, "history": []}
    return {"conversation_id": conv_id}


# ============================================================
# API 1: NHẬN DIỆN BỆNH
# ============================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = None,
):
    # Tạo session nếu chưa có
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    session = get_session(conversation_id)

    # Đọc và nhận diện ảnh
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File ảnh rỗng")

    try:
        disease = predict_disease(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi nhận diện: {str(e)}")

    # Lưu bệnh vào session
    session["disease"] = disease

    # Tạo prompt phân tích bệnh chi tiết
    analysis_prompt = build_disease_analysis_prompt(disease)

    # Thêm vào history: user gửi ảnh → AI phân tích
    session["history"].append({
        "role": "user",
        "content": f"[Người dùng đã chụp ảnh lá cây] AI nhận diện bệnh: {disease}\n\nHãy phân tích chi tiết bệnh này."
    })

    try:
        solution = ask_chatgpt(session["history"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi ChatGPT: {str(e)}")

    # Lưu phản hồi AI vào history
    session["history"].append({
        "role": "assistant",
        "content": solution
    })
    session["history"] = trim_history(session["history"])

    return {
        "conversation_id": conversation_id,
        "disease": disease,
        "solution": solution
    }


# ============================================================
# API 2: CHAT thường (fallback, không streaming)
# ============================================================

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    session = get_session(req.conversation_id)

    # Xây dựng user message — thêm context bệnh nếu có
    if session["disease"] and not _is_about_disease(req.question, session["disease"]):
        user_content = f"[Ngữ cảnh: cây đang mắc bệnh {session['disease']}]\n{req.question}"
    else:
        user_content = req.question

    session["history"].append({"role": "user", "content": user_content})

    try:
        answer = ask_chatgpt(session["history"])
    except Exception as e:
        session["history"].pop()  # Rollback nếu lỗi
        raise HTTPException(status_code=500, detail=f"Lỗi ChatGPT: {str(e)}")

    session["history"].append({"role": "assistant", "content": answer})
    session["history"] = trim_history(session["history"])

    return {
        "conversation_id": req.conversation_id,
        "answer": answer
    }


# ============================================================
# API 3: CHAT STREAMING (SSE)
# ============================================================

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    session = get_session(req.conversation_id)

    if session["disease"] and not _is_about_disease(req.question, session["disease"]):
        user_content = f"[Ngữ cảnh: cây đang mắc bệnh {session['disease']}]\n{req.question}"
    else:
        user_content = req.question

    session["history"].append({"role": "user", "content": user_content})

    # Chạy streaming và đồng thời tích lũy để lưu history
    history_snapshot = list(session["history"])

    async def event_generator():
        full_response = []
        try:
            for chunk in ask_chatgpt_stream(history_snapshot):
                full_response.append(chunk)
                # Encode JSON để giữ nguyên space, newline, ký tự đặc biệt
                # Flutter parse: json.decode(part) để lấy lại text gốc
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"

            # Lưu phản hồi đầy đủ vào history
            complete = "".join(full_response)
            session["history"].append({"role": "assistant", "content": complete})
            session["history"] = trim_history(session["history"])

        except Exception as e:
            yield f"data: {json.dumps('⚠️ Lỗi: ' + str(e), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            if session["history"]:
                session["history"].pop()  # Rollback

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================
# API 4: XOÁ SESSION (khi user xoá conversation)
# ============================================================

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if conversation_id in sessions:
        del sessions[conversation_id]
    return {"status": "deleted"}


# ============================================================
# HELPER
# ============================================================

def _is_about_disease(question: str, disease: str) -> bool:
    """Kiểm tra xem câu hỏi có liên quan đến bệnh đã chẩn đoán không."""
    q_lower = question.lower()
    d_lower = disease.lower()
    # Nếu người dùng nhắc tên bệnh hoặc hỏi chung về bệnh
    keywords = ["bệnh", "thuốc", "xử lý", "điều trị", "triệu chứng", "nguyên nhân"]
    return any(k in q_lower for k in keywords) or any(
        word in q_lower for word in d_lower.split()
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(sessions)}