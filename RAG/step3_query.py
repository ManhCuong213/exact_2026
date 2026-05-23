"""
BƯỚC 3: KẾT NỐI OLLAMA — ĐẶT CÂU HỎI, AI TRẢ LỜI
===================================================
Mục tiêu:
  - Nhận câu hỏi từ người dùng
  - Tìm chunks liên quan trong FAISS (Bước 2)
  - Gửi chunks + câu hỏi cho Ollama (qwen2.5:7b)
  - In câu trả lời ra terminal

Chạy: py step3_query.py
"""

import os
import requests
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==============================================================
# CẤU HÌNH
# ==============================================================
THU_MUC_INDEX = "./faiss_index"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL    = "http://localhost:11434/api/chat"  # Địa chỉ Ollama local
LLM_MODEL     = "qwen2.5:7b"                      # Model đang có
TOP_K         = 3                                  # Lấy 3 chunks liên quan nhất


# ==============================================================
# PHẦN 1: LOAD VECTOR DB (từ Bước 2)
# ==============================================================

def load_vector_db():
    """Load FAISS index đã tạo ở Bước 2."""
    print("[INFO] Đang load Vector DB...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_db = FAISS.load_local(
        THU_MUC_INDEX,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print(f"[OK] Load xong! Vectors trong DB: {vector_db.index.ntotal}\n")
    return vector_db


# Module-level cache — populated on first call, reused for all subsequent calls
_embeddings = None
_vector_db  = None


def _get_vector_db():
    global _embeddings, _vector_db
    if _vector_db is None:
        _dir       = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(_dir, "faiss_index")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vector_db = FAISS.load_local(
            index_path,
            _embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vector_db


def retrieve_context(question: str, question_type: str = "physics") -> str:
    """
    Retrieve relevant context from FAISS index for a given question.
    Returns top 3 chunks as a single string, or "" if nothing relevant found.
    """
    try:
        vector_db = _get_vector_db()

        # Physics scores cluster 0.5-0.7; logic scores cluster 0.87-0.95
        THRESHOLD = 0.75 if question_type == "physics" else 0.90

        results = vector_db.similarity_search_with_score(question, k=20)
        filtered = [doc for doc, score in results if score < THRESHOLD]
        top3 = filtered[:3]
        print(f"RAG [{question_type}]: {len(filtered)}/20 passed filter (threshold={THRESHOLD}), returning {len(top3)}")

        if not top3:
            return ""

        def _trim(text, limit=300):
            return text[:limit] + "..." if len(text) > limit else text
        return "\n---\n".join(_trim(doc.page_content) for doc in top3)

    except Exception:
        return ""


# ==============================================================
# PHẦN 2: KIỂM TRA OLLAMA CÓ ĐANG CHẠY KHÔNG
# ==============================================================

def kiem_tra_ollama():
    """
    Ping Ollama để xác nhận đang chạy.
    Ollama cần chạy nền trước khi gọi API.
    """
    try:
        res = requests.get("http://localhost:11434", timeout=3)
        if "Ollama" in res.text:
            print(f"[OK] Ollama đang chạy — model: {LLM_MODEL}\n")
            return True
    except Exception:
        pass

    print("[!] Ollama chưa chạy!")
    print("    Mở terminal mới và chạy: ollama serve")
    print("    Sau đó chạy lại file này.\n")
    return False


# ==============================================================
# PHẦN 3: TÌM CHUNKS LIÊN QUAN (Retrieve)
# ==============================================================

def tim_chunks_lien_quan(cau_hoi, vector_db, top_k=3):
    """
    Tìm top_k chunks liên quan nhất với câu hỏi trong Vector DB.

    Luồng bên trong:
      1. Chuyển câu hỏi → vector (dùng embedding model)
      2. So sánh với toàn bộ vectors trong FAISS
      3. Trả về top_k chunks có vector gần nhất
    """
    ket_qua = vector_db.similarity_search(cau_hoi, k=top_k)
    return ket_qua


# ==============================================================
# PHẦN 4: GỌI OLLAMA (Generate)
# ==============================================================
#
# Luồng RAG hoàn chỉnh:
#   Câu hỏi → Tìm chunks liên quan → Ghép thành context
#   → Gửi [context + câu hỏi] cho Ollama → Nhận câu trả lời
#
# Tại sao không gửi thẳng câu hỏi cho Ollama?
#   Vì Ollama không biết gì về datasheet của bạn.
#   Phải đưa context (đoạn trích từ datasheet) vào prompt
#   thì Ollama mới có thông tin để trả lời đúng.

def goi_ollama(cau_hoi, chunks):
    """
    Ghép context từ chunks + câu hỏi thành prompt,
    gửi cho Ollama và nhận câu trả lời.
    """

    # Ghép nội dung các chunks thành 1 đoạn context
    context = "\n\n---\n\n".join(
        f"Nguồn: {os.path.basename(doc.metadata.get('source', 'N/A'))}\n"
        f"{doc.page_content}"
        for doc in chunks
    )

    # Prompt gửi cho AI
    # Cấu trúc: [Vai trò] + [Context] + [Câu hỏi] + [Yêu cầu trả lời]
    prompt = f"""Bạn là trợ lý kỹ thuật phân tích datasheet linh kiện điện tử.
Dựa vào các đoạn datasheet dưới đây, hãy trả lời câu hỏi của người dùng.
Nếu không tìm thấy thông tin trong datasheet, hãy nói rõ là không có dữ liệu.

=== DATASHEET CONTEXT ===
{context}

=== CÂU HỎI ===
{cau_hoi}

=== TRẢ LỜI ==="""

    # Gọi Ollama API
    # Ollama chạy local tại http://localhost:11434
    # Không cần API key, không gửi dữ liệu ra ngoài
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,   # False = chờ xong mới trả về, dễ xử lý hơn
            },
            timeout=120,           # Chờ tối đa 2 phút
        )

        data = response.json()
        tra_loi = data["message"]["content"]
        return tra_loi

    except requests.exceptions.Timeout:
        return "[LỖI] Ollama xử lý quá lâu (>2 phút). Thử câu hỏi ngắn hơn."
    except Exception as loi:
        return f"[LỖI] Không gọi được Ollama: {loi}"


# ==============================================================
# PHẦN 5: VÒNG LẶP HỎI-ĐÁP
# ==============================================================

def chay_vong_lap_hoi_dap(vector_db):
    """Vòng lặp chính: nhận câu hỏi → tìm kiếm → gọi AI → in kết quả."""

    print("="*55)
    print("  HỎI-ĐÁP VỀ DATASHEET (dùng qwen2.5:7b)")
    print("  Gõ câu hỏi bất kỳ. Gõ 'exit' để thoát.")
    print("="*55)

    while True:
        print()
        cau_hoi = input("Câu hỏi của bạn: ").strip()

        if cau_hoi.lower() == "exit":
            print("[BYE] Thoát chương trình.")
            break

        if not cau_hoi:
            continue

        print(f"\n[1/3] Đang tìm chunks liên quan trong Vector DB...")
        chunks = tim_chunks_lien_quan(cau_hoi, vector_db, TOP_K)

        print(f"[2/3] Tìm thấy {len(chunks)} chunks:")
        for i, doc in enumerate(chunks):
            ten = os.path.basename(doc.metadata.get("source", "N/A"))
            print(f"      #{i+1} {ten} — {doc.page_content[:80].strip()}...")

        print(f"[3/3] Đang gửi cho Ollama ({LLM_MODEL}), chờ trả lời...\n")
        tra_loi = goi_ollama(cau_hoi, chunks)

        print("="*55)
        print("TRẢ LỜI:")
        print("="*55)
        print(tra_loi)
        print("="*55)


# ==============================================================
# CHẠY CHƯƠNG TRÌNH
# ==============================================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  BƯỚC 3: RAG + OLLAMA")
    print("="*55 + "\n")

    # Kiểm tra Ollama
    if not kiem_tra_ollama():
        exit()

    # Load Vector DB
    vector_db = load_vector_db()

    # Bắt đầu hỏi-đáp
    chay_vong_lap_hoi_dap(vector_db)
