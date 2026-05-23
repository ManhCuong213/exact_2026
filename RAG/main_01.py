"""
PIPELINE HOÀN CHỈNH: BƯỚC 1 → BƯỚC 2 → BƯỚC 3
================================================
Chạy: python main.py

Luồng dữ liệu:
  PDF files
    ↓  (Bước 1) Đọc & Chunk
  chunks
    ↓  (Bước 2) Embedding → FAISS
  vector_db
    ↓  (Bước 3) Ollama RAG
  Hỏi-đáp terminal
"""

# ── Import từ từng bước (không chạy lại code, chỉ dùng hàm) ──
from step1_chunking import tao_pdf_gia, xu_ly_tat_ca_pdf, THU_MUC_PDF, BATCH_SIZE
from step2_vectordb import load_embedding_model, load_hoac_tao_faiss, test_tim_kiem, THU_MUC_INDEX
from step3_query    import kiem_tra_ollama, chay_vong_lap_hoi_dap


def main():
    print("\n" + "=" * 58)
    print("  PIPELINE: BƯỚC 1 → BƯỚC 2 → BƯỚC 3")
    print("=" * 58 + "\n")

    # ──────────────────────────────────────────────────────────
    # BƯỚC 1: Đọc PDF → Chunk
    # ──────────────────────────────────────────────────────────
    tao_pdf_gia(THU_MUC_PDF, so_luong=20)          # Bỏ dòng này nếu đã có PDF thật
    chunks = xu_ly_tat_ca_pdf(THU_MUC_PDF, batch_size=BATCH_SIZE)

    if not chunks:
        print("[!] Không có chunks. Hãy chép file PDF vào './pdfs/' rồi chạy lại.")
        return

    # ──────────────────────────────────────────────────────────
    # BƯỚC 2: Embedding + FAISS — nhận chunks trực tiếp từ Bước 1
    # ──────────────────────────────────────────────────────────
    embeddings = load_embedding_model()
    vector_db  = load_hoac_tao_faiss(chunks, embeddings, THU_MUC_INDEX)
    test_tim_kiem(vector_db)

    # ──────────────────────────────────────────────────────────
    # BƯỚC 3: Ollama RAG — nhận vector_db trực tiếp từ Bước 2
    # (không load lại từ disk, dùng ngay object đang có trong RAM)
    # ──────────────────────────────────────────────────────────
    if not kiem_tra_ollama():
        print("[!] Hãy mở terminal khác, chạy:  ollama serve")
        print("    Sau đó chạy lại:              python main.py")
        return

    chay_vong_lap_hoi_dap(vector_db)


if __name__ == "__main__":
    main()
