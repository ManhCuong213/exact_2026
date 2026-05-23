"""
BƯỚC 2: LƯU TRỮ VÀO VECTOR DATABASE (Embeddings + FAISS)
==========================================================
Mục tiêu:
  - Nhận chunks từ Bước 1
  - Chuyển mỗi chunk thành vector số (Embeddings)
  - Lưu toàn bộ vào FAISS (Vector Database chạy local)
  - Test thử tìm kiếm để xác nhận hoạt động đúng

Chạy: py step2_vectordb.py
"""

import os
import glob
import pickle

# ---- Thư viện từ Bước 1 (dùng lại) ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Thư viện mới cho Bước 2 ----
# HuggingFaceEmbeddings: chuyển text → vector số (chạy local, không cần API)
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS: Vector Database lưu và tìm kiếm vector nhanh
from langchain_community.vectorstores import FAISS


# ==============================================================
# CẤU HÌNH
# ==============================================================
THU_MUC_PDF    = "./pdfs"           # Thư mục chứa file PDF (từ Bước 1)
THU_MUC_INDEX  = "./faiss_index"    # Nơi lưu Vector Database xuống disk
CHUNK_SIZE     = 500                # Giữ nguyên như Bước 1
CHUNK_OVERLAP  = 50                 # Giữ nguyên như Bước 1
BATCH_SIZE     = 50                 # Số file xử lý mỗi lần

# Model tạo Embeddings (chạy hoàn toàn local, không cần internet sau lần đầu)
# all-MiniLM-L6-v2: nhỏ (~90MB), nhanh, phù hợp cho PoC
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"


# ==============================================================
# PHẦN 1: LẶP LẠI BƯỚC 1 (đọc PDF → chunks)
# ==============================================================
# Lý do lặp lại: Bước 1 và Bước 2 cần chạy liên tiếp trong cùng 1 pipeline
# Sau này sẽ gộp thành 1 file duy nhất

def doc_va_chunk_pdf(thu_muc, batch_size=50):
    """Đọc toàn bộ PDF và trả về danh sách chunks (giống Bước 1)."""

    tat_ca_file = sorted(glob.glob(f"{thu_muc}/*.pdf"))
    tong_so     = len(tat_ca_file)

    if tong_so == 0:
        print(f"[!] Không tìm thấy file PDF trong '{thu_muc}/'")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    print(f"\n{'='*58}")
    print(f"  BƯỚC 1 (nhanh): ĐỌC & CHUNK {tong_so} FILE PDF")
    print(f"{'='*58}")

    tat_ca_chunks = []

    for bat_dau in range(0, tong_so, batch_size):
        ket_thuc    = min(bat_dau + batch_size, tong_so)
        batch_files = tat_ca_file[bat_dau:ket_thuc]

        for i, duong_dan in enumerate(batch_files):
            chi_so   = bat_dau + i + 1
            ten_file = os.path.basename(duong_dan)
            print(f"  [INFO] Đang đọc file {chi_so}/{tong_so} — {ten_file}")

            try:
                loader = PyPDFLoader(duong_dan)
                pages  = loader.load()
                chunks = splitter.split_documents(pages)
                tat_ca_chunks.extend(chunks)
                print(f"         → {len(pages)} trang | {len(chunks)} chunks")
            except Exception as loi:
                print(f"  [!] Lỗi đọc file {ten_file}: {loi}")
                continue

    print(f"\n[OK] Tổng chunks: {len(tat_ca_chunks)}\n")
    return tat_ca_chunks


# ==============================================================
# PHẦN 2: TẠO EMBEDDINGS (chuyển text → vector số)
# ==============================================================
#
# Embedding là gì?
# ─────────────────
# Máy tính không hiểu chữ, chỉ hiểu số.
# Embedding chuyển 1 đoạn text thành 1 dãy số (vector).
#
# Ví dụ:
#   "Resistor 10kΩ"  → [0.21, -0.54, 0.87, 0.12, ...]  (384 số)
#   "Tụ điện 100uF"  → [0.18, -0.51, 0.79, 0.09, ...]  (384 số)
#   "Mèo đang ngủ"   → [0.95,  0.34, -0.22, 0.67, ...]  (384 số)
#
# 2 đoạn text nói về chủ đề giống nhau → vector gần nhau trong không gian số
# 2 đoạn text khác chủ đề hoàn toàn   → vector xa nhau
#
# Đây là nền tảng để "tìm kiếm theo nghĩa" thay vì tìm từ khóa

def load_embedding_model():
    """
    Tải model Embedding về máy (lần đầu ~90MB, lần sau dùng cache).
    Model chạy hoàn toàn local — không gửi dữ liệu ra ngoài.
    """
    print(f"{'='*58}")
    print(f"  BƯỚC 2A: TẢI EMBEDDING MODEL")
    print(f"{'='*58}")
    print(f"[INFO] Model: {EMBED_MODEL}")
    print(f"[INFO] Lần đầu chạy sẽ tải ~90MB về máy, chờ 1-2 phút...")
    print(f"[INFO] Lần sau sẽ load từ cache, rất nhanh.\n")

    # Tạo đối tượng embeddings
    # model_kwargs: chạy trên CPU (không cần GPU)
    # encode_kwargs: chuẩn hóa vector để tính khoảng cách chính xác hơn
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[OK] Load model thành công!\n")
    return embeddings


# ==============================================================
# PHẦN 3: XÂY DỰNG FAISS INDEX (lưu vector vào DB)
# ==============================================================
#
# FAISS là gì?
# ─────────────
# FAISS = Facebook AI Similarity Search
# Là một thư viện lưu trữ hàng triệu vector và tìm kiếm cực nhanh.
#
# Hoạt động như thế nào?
#   1. Nhận vào: danh sách chunks + embedding model
#   2. Chuyển mỗi chunk thành vector (bằng embedding model)
#   3. Lưu tất cả vector vào một "kho" được tổ chức để tìm kiếm nhanh
#
# Khi query:
#   - Câu hỏi → chuyển thành vector
#   - FAISS tìm 5 vector gần nhất trong kho
#   - Trả về 5 chunks tương ứng

def xay_dung_faiss_index(chunks, embeddings, thu_muc_luu):
    """
    Nhận chunks + embedding model → tạo FAISS index → lưu xuống disk.
    """
    print(f"{'='*58}")
    print(f"  BƯỚC 2B: XÂY DỰNG FAISS VECTOR DATABASE")
    print(f"{'='*58}")
    print(f"[INFO] Số chunks cần embed: {len(chunks)}")
    print(f"[INFO] Đang chuyển {len(chunks)} chunks thành vectors...")
    print(f"       (Mỗi chunk → 384 con số)")
    print(f"       Chờ 1-3 phút tùy tốc độ máy...\n")

    # FAISS.from_documents() làm 2 việc cùng lúc:
    #   1. Gọi embeddings.embed_documents() để chuyển mọi chunk → vector
    #   2. Lưu toàn bộ vector vào FAISS index
    #
    # LangChain tự động xử lý theo batch nhỏ bên trong → không lo OOM
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    print(f"[OK] Tạo FAISS index thành công!")
    print(f"     Tổng vectors trong DB: {vector_db.index.ntotal}\n")

    # Lưu xuống disk để lần sau không phải tạo lại
    # Sẽ tạo 2 file: index.faiss và index.pkl
    os.makedirs(thu_muc_luu, exist_ok=True)
    vector_db.save_local(thu_muc_luu)

    print(f"[OK] Đã lưu Vector DB tại '{thu_muc_luu}/'")
    print(f"     → {thu_muc_luu}/index.faiss  (các vectors)")
    print(f"     → {thu_muc_luu}/index.pkl    (nội dung chunks)\n")

    return vector_db


# ==============================================================
# PHẦN 4: LOAD LẠI FAISS (nếu đã tạo rồi thì không tạo lại)
# ==============================================================

def load_hoac_tao_faiss(chunks, embeddings, thu_muc_luu):
    """
    Kiểm tra xem FAISS index đã có chưa:
    - Có rồi → load từ disk (nhanh, vài giây)
    - Chưa có → tạo mới (chậm hơn, 1-3 phút)
    """
    file_index = os.path.join(thu_muc_luu, "index.faiss")

    if os.path.exists(file_index):
        print(f"[INFO] Phát hiện FAISS index đã có tại '{thu_muc_luu}/'")
        print(f"[INFO] Load từ disk, không cần tạo lại...\n")

        # allow_dangerous_deserialization=True: cần thiết với FAISS mới
        vector_db = FAISS.load_local(
            thu_muc_luu,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[OK] Load thành công! Vectors trong DB: {vector_db.index.ntotal}\n")
        return vector_db

    else:
        print(f"[INFO] Chưa có FAISS index, bắt đầu tạo mới...\n")
        return xay_dung_faiss_index(chunks, embeddings, thu_muc_luu)


# ==============================================================
# PHẦN 5: TEST THỬ TÌM KIẾM
# ==============================================================
#
# Đây là bước quan trọng để xác nhận Vector DB hoạt động đúng.
# Thay vì chờ đến Bước 3 mới biết có lỗi, test ngay tại đây.

def test_tim_kiem(vector_db):
    """
    Thử tìm kiếm 2 câu hỏi mẫu trong Vector DB.
    Mục tiêu: xác nhận DB trả về kết quả có liên quan.
    """
    print(f"{'='*58}")
    print(f"  BƯỚC 2C: TEST TÌM KIẾM TRONG VECTOR DB")
    print(f"{'='*58}")
    print(f"[INFO] Chạy 2 câu hỏi thử để xác nhận DB hoạt động...\n")

    cau_hoi_mau = [
        "Resistor có điện áp bao nhiêu?",
        "Linh kiện nào của Texas Instruments?",
    ]

    for i, cau_hoi in enumerate(cau_hoi_mau):
        print(f"  [QUERY #{i+1}] {cau_hoi}")
        print(f"  {'─'*50}")

        # similarity_search: tìm k chunks giống câu hỏi nhất
        # Bên trong nó làm:
        #   1. Chuyển câu hỏi → vector
        #   2. So sánh với toàn bộ vectors trong DB
        #   3. Trả về k chunks có vector gần nhất
        ket_qua = vector_db.similarity_search(cau_hoi, k=3)

        for j, doc in enumerate(ket_qua):
            ten_file = os.path.basename(doc.metadata.get("source", "N/A"))
            trang    = doc.metadata.get("page", "N/A")
            noi_dung = doc.page_content[:150].replace("\n", " ")

            print(f"  Kết quả #{j+1}:")
            print(f"    File   : {ten_file} (trang {trang})")
            print(f"    Nội dung: {noi_dung}...")
            print()

    print(f"[OK] Test tìm kiếm thành công!")
    print(f"     Vector DB đang hoạt động đúng — sẵn sàng cho Bước 3\n")


# ==============================================================
# CHẠY CHƯƠNG TRÌNH
# ==============================================================

if __name__ == "__main__":

    print("\n" + "="*58)
    print("  BƯỚC 2: XÂY DỰNG VECTOR DATABASE")
    print("  (Embeddings + FAISS)")
    print("="*58)

    # --- 2.0: Đọc lại chunks từ Bước 1 ---
    chunks = doc_va_chunk_pdf(THU_MUC_PDF, BATCH_SIZE)
    if not chunks:
        print("[!] Không có chunks để xử lý. Hãy chạy Bước 1 trước.")
        exit()

    # --- 2A: Tải embedding model ---
    embeddings = load_embedding_model()

    # --- 2B: Tạo hoặc load FAISS index ---
    vector_db = load_hoac_tao_faiss(chunks, embeddings, THU_MUC_INDEX)

    # --- 2C: Test thử tìm kiếm ---
    test_tim_kiem(vector_db)

    print("="*58)
    print("  BƯỚC 2 HOÀN THÀNH!")
    print("="*58)
    print(f"""
  Tóm tắt:
  ─────────────────────────────────────────
  ✓ Đã tạo embeddings cho {len(chunks)} chunks
  ✓ Đã lưu FAISS index tại '{THU_MUC_INDEX}/'
  ✓ Test tìm kiếm thành công

  Tiếp theo → Bước 3: Gọi Ollama để trả lời câu hỏi
  ─────────────────────────────────────────
    """)
