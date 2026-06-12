"""
rag_service.py
──────────────
Hệ thống RAG (Retrieval-Augmented Generation) cho Plant Doctor AI.

Cấu trúc thư mục tài liệu:
    docs/
    ├── lua/          ← PDF/TXT về bệnh lúa
    ├── ca_phe/       ← PDF/TXT về bệnh cà phê
    └── tieu/         ← PDF/TXT về bệnh hồ tiêu

Workflow:
    1. build_all_indexes()  ← chạy 1 lần khi khởi động server
    2. retrieve_context(disease, query) ← gọi mỗi khi cần trả lời

Cài đặt:
    pip install langchain langchain-community langchain-openai faiss-cpu pypdf sentence-transformers
"""

import os
import json
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────────

DOCS_ROOT = Path("docs")          # thư mục gốc chứa tài liệu
INDEX_ROOT = Path("faiss_indexes") # thư mục lưu FAISS index

# Ánh xạ: tên cây → thư mục tài liệu
CROP_FOLDERS: dict[str, str] = {
    "lua":    "lua",
    "ca_phe": "ca_phe",
    "tieu":   "tieu",
}

# Từ khóa nhận diện loại cây từ tên bệnh
CROP_KEYWORDS: dict[str, list[str]] = {
    "lua": [
        "lua", "lúa", "rice", "oryza",
        "dao on", "đạo ôn", "bạc lá", "bac la",
        "lem lep", "lép", "lem léo", "tung hung",
    ],
    "ca_phe": [
        "ca phe", "cà phê", "coffee", "coffea",
        "gi sat", "gỉ sắt", "rust",
        "kho canh", "khô cành", "tuyến trùng",
    ],
    "tieu": [
        "tieu", "tiêu", "pepper", "piper",
        "chet nhanh", "chết nhanh", "thoi re", "thối rễ",
        "lep trai", "lép trái",
    ],
}

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

# Cache các vector store đã load
_vector_stores: dict[str, FAISS] = {}


# ── BUILD INDEX ────────────────────────────────────────────────────────────────

def build_index_for_crop(crop_key: str) -> None:
    """
    Đọc toàn bộ PDF/TXT trong docs/<crop_key>/, chunk và tạo FAISS index.
    Index được lưu vào faiss_indexes/<crop_key>/ để dùng lại.
    """
    folder = DOCS_ROOT / CROP_FOLDERS[crop_key]
    index_path = INDEX_ROOT / crop_key

    if not folder.exists():
        print(f"[RAG] Thư mục {folder} không tồn tại, bỏ qua.")
        return

    # Nếu index đã tồn tại thì load luôn, không build lại
    if index_path.exists():
        print(f"[RAG] Load index sẵn có: {index_path}")
        _vector_stores[crop_key] = FAISS.load_local(
            str(index_path), EMBEDDINGS, allow_dangerous_deserialization=True
        )
        return

    print(f"[RAG] Building index cho {crop_key} từ {folder} ...")

    docs: list[Document] = []
    for file_path in sorted(folder.glob("**/*")):
        if not file_path.is_file():
            continue
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in (".txt", ".md"):
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                continue

            loaded = loader.load()
            # Thêm metadata: tên file, trang, crop
            for doc in loaded:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["crop"]        = crop_key
            docs.extend(loaded)
        except Exception as e:
            print(f"[RAG] Lỗi đọc {file_path.name}: {e}")

    if not docs:
        print(f"[RAG] Không có tài liệu nào cho {crop_key}")
        return

    # Chunk tài liệu
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"[RAG] {crop_key}: {len(docs)} docs → {len(chunks)} chunks")

    # Build FAISS
    vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)

    # Lưu xuống disk
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    _vector_stores[crop_key] = vectorstore
    print(f"[RAG] Đã lưu index: {index_path}")


def build_all_indexes() -> None:
    """Khởi tạo tất cả FAISS indexes khi server start."""
    INDEX_ROOT.mkdir(exist_ok=True)
    for crop_key in CROP_FOLDERS:
        build_index_for_crop(crop_key)


# ── DETECT CROP TYPE ───────────────────────────────────────────────────────────

def detect_crop_from_disease(disease_name: str) -> Optional[str]:
    """
    Phát hiện loại cây từ tên bệnh.
    Trả về 'lua' | 'ca_phe' | 'tieu' | None
    """
    disease_lower = disease_name.lower().replace("_", " ")
    for crop_key, keywords in CROP_KEYWORDS.items():
        for kw in keywords:
            if kw in disease_lower:
                return crop_key
    return None


# ── RETRIEVE CONTEXT ───────────────────────────────────────────────────────────

def retrieve_context(
    disease_name: str,
    user_query: str = "",
    top_k: int = 3,
) -> dict:
    """
    Tìm kiếm trong vector store các đoạn tài liệu liên quan nhất.

    Returns
    -------
    dict:
        crop        : str | None   – loại cây phát hiện được
        chunks      : list[dict]   – mỗi chunk có {content, source_file, page, score}
        context_str : str          – chuỗi ghép lại để nhét vào prompt
        citations   : list[str]    – danh sách trích dẫn ngắn gọn
    """
    crop_key = detect_crop_from_disease(disease_name)

    if crop_key is None or crop_key not in _vector_stores:
        # Không có tài liệu phù hợp — trả về rỗng, LLM tự trả lời
        return {
            "crop": crop_key,
            "chunks": [],
            "context_str": "",
            "citations": [],
        }

    vectorstore = _vector_stores[crop_key]

    # Câu truy vấn kết hợp tên bệnh + câu hỏi người dùng
    search_query = f"bệnh {disease_name} {user_query}".strip()

    results = vectorstore.similarity_search_with_score(search_query, k=top_k)

    chunks = []
    citations = []
    context_parts = []

    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page   = doc.metadata.get("page", "?")
        content = doc.page_content.strip()

        chunks.append({
            "content":     content,
            "source_file": source,
            "page":        page,
            "score":       round(float(score), 4),
        })

        citation = f"[{i}] {source}, trang {page}"
        citations.append(citation)

        context_parts.append(
            f"--- Tài liệu {i} ({source}, trang {page}) ---\n{content}"
        )

    context_str = "\n\n".join(context_parts)

    return {
        "crop":        crop_key,
        "chunks":      chunks,
        "context_str": context_str,
        "citations":   citations,
    }


# ── BUILD PROMPT RAG ───────────────────────────────────────────────────────────

def build_rag_prompt(
    disease: str,
    user_question: str,
    rag_result: dict,
    confidence: float = 1.0,
) -> str:
    """
    Tạo prompt đầy đủ kết hợp ngữ cảnh RAG + thông tin bệnh.
    Prompt này được gửi đi thay cho build_disease_analysis_prompt() cũ.
    """
    crop_names = {
        "lua":    "lúa",
        "ca_phe": "cà phê",
        "tieu":   "hồ tiêu",
    }
    crop_display = crop_names.get(rag_result.get("crop", ""), "cây trồng")
    citations_text = "\n".join(rag_result.get("citations", [])) or "Không có tài liệu tham khảo."

    rag_context = rag_result.get("context_str", "")

    if rag_context:
        context_section = f"""
📚 TÀI LIỆU THAM KHẢO (trích từ cơ sở dữ liệu {crop_display}):
{rag_context}

Nguồn trích dẫn:
{citations_text}
"""
    else:
        context_section = "(Không có tài liệu tham khảo cho bệnh này trong cơ sở dữ liệu.)"

    prompt = f"""Cây {crop_display} vừa được AI chẩn đoán mắc bệnh: **{disease}**
Độ tin cậy của mô hình: {confidence:.1%}

{context_section}

Dựa trên tài liệu trên (nếu có) và kiến thức chuyên môn, hãy trả lời theo cấu trúc:

🔬 **Bệnh:** {disease}

🌱 **Nguyên nhân:** (giải thích ngắn gọn, ưu tiên dùng thông tin từ tài liệu)

⚠️ **Dấu hiệu nhận biết:** (triệu chứng trên lá, thân, quả)

💊 **Cách điều trị:** (bước cụ thể, tên thuốc nếu có, thời gian phun)

🛡️ **Phòng tránh:** (biện pháp ngăn ngừa tái phát)

⏰ **Mức độ khẩn cấp:** Thấp / Trung bình / Cao

📖 **Nguồn tham khảo:** (liệt kê tài liệu đã dùng, ví dụ: "Theo tài liệu [1] trang X...")

Câu hỏi bổ sung của người dùng: {user_question if user_question else "(không có)"}
"""
    return prompt