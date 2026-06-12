"""
app.py  (v3.0 — RAG + GradCAM)
──────────────────────────────
Các thay đổi so với v2.0:
  • /detect   : thêm GradCAM heatmap + RAG context
  • /chat     : RAG retrieve trước khi gọi LLM
  • /rebuild  : (admin) rebuild lại FAISS indexes
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json
import os

from predict1 import predict_disease, predict_disease_detailed
from chatgpt_service import ask_chatgpt, ask_chatgpt_stream, validate_plant_image
from gradcam_service import generate_gradcam
from rag_service import build_all_indexes, retrieve_context, build_rag_prompt

# ── Import model để truyền vào GradCAM ────────────────────────────────────────
import torch
import timm

def _load_model():
    ckpt = torch.load("best_mobilevit.pth", map_location="cpu")
    classes = ckpt["classes"]
    model = timm.create_model("mobilevit_s", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes

_model, _classes = _load_model()


# ── LIFESPAN: khởi động RAG indexes ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Building RAG indexes ...")
    build_all_indexes()
    print("[Startup] Ready!")
    yield

app = FastAPI(title="Plant Doctor AI", version="3.0", lifespan=lifespan)


# ── SESSION STORE ──────────────────────────────────────────────────────────────

sessions: dict[str, dict] = {}
MAX_HISTORY = 20

def get_session(conversation_id: str) -> dict:
    if conversation_id not in sessions:
        sessions[conversation_id] = {
            "disease": None,
            "crop":    None,
            "history": [],
        }
    return sessions[conversation_id]

def trim_history(history: list) -> list:
    return history[-MAX_HISTORY:] if len(history) > MAX_HISTORY else history


# ── MODELS ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    conversation_id: str
    question: str

class NewConversationResponse(BaseModel):
    conversation_id: str


# ── API: Tạo conversation mới ──────────────────────────────────────────────────

@app.post("/conversation/new", response_model=NewConversationResponse)
async def new_conversation():
    conv_id = str(uuid.uuid4())
    sessions[conv_id] = {"disease": None, "crop": None, "history": []}
    return {"conversation_id": conv_id}


# ── API 1: NHẬN DIỆN BỆNH + GRADCAM + RAG ─────────────────────────────────────

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = None,
):
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    session = get_session(conversation_id)
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="File ảnh rỗng")

    # ── BƯỚC 1: Validate ảnh ──────────────────────────────────────────────────
    try:
        validation = validate_plant_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi kiểm tra ảnh: {e}")

    if not validation["is_plant"]:
        return {
            "conversation_id": conversation_id,
            "disease": None,
            "heatmap_b64": None,
            "bbox": None,
            "coverage_pct": None,
            "solution": validation["message"],
            "citations": [],
        }

    # ── BƯỚC 2: Predict bệnh (chi tiết) ──────────────────────────────────────
    try:
        detail = predict_disease_detailed(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi nhận diện: {e}")

    disease    = detail["disease"]
    confidence = detail["confidence"]
    session["disease"] = disease

    # ── BƯỚC 3: GradCAM heatmap ───────────────────────────────────────────────
    gradcam_result = {}
    try:
        gradcam_result = generate_gradcam(
            image_bytes=image_bytes,
            model=_model,
            classes=_classes,
            target_idx=None,   # tự dùng lớp có score cao nhất
        )
    except Exception as e:
        print(f"[GradCAM] Lỗi: {e}")
        gradcam_result = {"heatmap_b64": None, "bbox": None, "coverage_pct": None}

    # ── BƯỚC 4: RAG — tìm tài liệu liên quan ─────────────────────────────────
    rag_result = retrieve_context(disease_name=disease, user_query="", top_k=3)
    session["crop"] = rag_result.get("crop")

    # ── BƯỚC 5: Build prompt có RAG + gọi LLM ────────────────────────────────
    rag_prompt = build_rag_prompt(
        disease=disease,
        user_question="",
        rag_result=rag_result,
        confidence=confidence,
    )

    session["history"].append({
        "role": "user",
        "content": f"[Người dùng đã chụp ảnh lá cây]\n{rag_prompt}",
    })

    try:
        solution = ask_chatgpt(session["history"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi ChatGPT: {e}")

    session["history"].append({"role": "assistant", "content": solution})
    session["history"] = trim_history(session["history"])

    return {
        "conversation_id": conversation_id,
        "disease":         disease,
        "confidence":      confidence,
        "top3":            detail.get("top3", []),
        # GradCAM
        "heatmap_b64":     gradcam_result.get("heatmap_b64"),
        "bbox":            gradcam_result.get("bbox"),
        "coverage_pct":    gradcam_result.get("coverage_pct"),
        # RAG
        "crop":            rag_result.get("crop"),
        "citations":       rag_result.get("citations", []),
        # Giải pháp
        "solution":        solution,
    }


# ── API 2: CHAT STREAMING với RAG ─────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    session = get_session(req.conversation_id)
    disease = session.get("disease")

    # RAG retrieve dựa trên bệnh đang active trong session
    rag_result = {}
    if disease:
        rag_result = retrieve_context(
            disease_name=disease,
            user_query=req.question,
            top_k=2,
        )

    # Nếu có tài liệu RAG, nhúng vào system context; nếu không thì gửi thẳng
    if rag_result.get("context_str"):
        user_content = (
            f"[Ngữ cảnh tài liệu tham khảo]\n{rag_result['context_str']}\n\n"
            f"[Câu hỏi] {req.question}"
        )
    elif disease:
        user_content = f"[Cây đang mắc bệnh {disease}]\n{req.question}"
    else:
        user_content = req.question

    session["history"].append({"role": "user", "content": user_content})
    history_snapshot = list(session["history"])

    async def event_generator():
        full_response = []
        try:
            for chunk in ask_chatgpt_stream(history_snapshot):
                full_response.append(chunk)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

            complete = "".join(full_response)
            session["history"].append({"role": "assistant", "content": complete})
            session["history"] = trim_history(session["history"])

        except Exception as e:
            yield f"data: {json.dumps('⚠️ Lỗi: ' + str(e), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            if session["history"]:
                session["history"].pop()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── API 3: CHAT thường (fallback) ─────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    session = get_session(req.conversation_id)
    disease = session.get("disease")

    rag_result = {}
    if disease:
        rag_result = retrieve_context(disease_name=disease, user_query=req.question, top_k=2)

    if rag_result.get("context_str"):
        user_content = (
            f"[Tài liệu tham khảo]\n{rag_result['context_str']}\n\n"
            f"[Câu hỏi] {req.question}"
        )
    else:
        user_content = req.question

    session["history"].append({"role": "user", "content": user_content})

    try:
        answer = ask_chatgpt(session["history"])
    except Exception as e:
        session["history"].pop()
        raise HTTPException(status_code=500, detail=f"Lỗi ChatGPT: {e}")

    session["history"].append({"role": "assistant", "content": answer})
    session["history"] = trim_history(session["history"])

    return {
        "conversation_id": req.conversation_id,
        "answer": answer,
        "citations": rag_result.get("citations", []),
    }


# ── API 4: Rebuild RAG indexes (admin) ────────────────────────────────────────

@app.post("/admin/rebuild-indexes")
async def rebuild_indexes(secret: str = ""):
    if secret != os.getenv("ADMIN_SECRET", "plant123"):
        raise HTTPException(status_code=403, detail="Sai secret key")
    build_all_indexes()
    return {"status": "rebuilt"}


# ── API 5: Xoá session ────────────────────────────────────────────────────────

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if conversation_id in sessions:
        del sessions[conversation_id]
    return {"status": "deleted"}


# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions_active": len(sessions),
        "rag_loaded": list(_vector_stores_loaded()),
    }

def _vector_stores_loaded():
    from rag_service import _vector_stores
    return _vector_stores.keys()

