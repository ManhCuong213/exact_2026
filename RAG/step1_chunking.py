"""
BƯỚC 1: XỬ LÝ TÀI LIỆU (Document Loading & Chunking)
======================================================
Mục tiêu: Đọc file PDF → Cắt thành đoạn nhỏ (chunks) → In ra màn hình

Chạy: python step1_chunking.py
"""

import os
import glob
import sys

# LangChain: thư viện giúp làm việc với AI và tài liệu
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Excel / CSV loader từ pdfs/excel_loader.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs"))
from excel_loader import excel_to_documents, csv_to_documents

# ==============================================================
# PHẦN 1: CẤU HÌNH
# ==============================================================

# Thư mục chứa các file PDF
THU_MUC_PDF = "./pdfs"

# Thư mục chứa Excel và CSV (đặt trong pdfs/document/)
THU_MUC_DOCUMENT = "./pdfs/document"

# Mỗi lần xử lý bao nhiêu file? (50 = an toàn cho RAM 16GB)
BATCH_SIZE = 50

# Mỗi chunk tối đa bao nhiêu ký tự?
# (500 ký tự ≈ 1 đoạn văn ngắn — vừa đủ để AI hiểu ngữ cảnh)
CHUNK_SIZE = 500

# Cho phép 2 chunk liền kề "chồng lấp" nhau bao nhiêu ký tự?
# (50 ký tự = giúp không bị mất thông tin ở ranh giới giữa 2 chunk)
CHUNK_OVERLAP = 50


# ==============================================================
# PHẦN 2: TẠO FILE PDF GIẢ ĐỂ TEST (nếu chưa có file thật)
# ==============================================================

def tao_pdf_gia(thu_muc, so_luong=10):
    """
    Tạo file PDF giả để test — bỏ hàm này khi dùng file thật.
    Cần cài: pip install reportlab
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
    except ImportError:
        print("[!] Chưa cài reportlab. Chạy: pip install reportlab")
        print("    Hoặc chép file PDF thật vào thư mục ./pdfs/")
        return

    os.makedirs(thu_muc, exist_ok=True)
    da_co = glob.glob(f"{thu_muc}/*.pdf")

    if len(da_co) >= so_luong:
        print(f"[OK] Đã có {len(da_co)} file PDF, bỏ qua bước tạo file giả.")
        return

    print(f"[SETUP] Tạo {so_luong} file PDF giả để test...")

    linh_kien = ["Resistor", "Capacitor", "Transistor", "Diode", "LED"]

    for i in range(len(da_co), so_luong):
        ten_file = f"{thu_muc}/datasheet_{i+1:04d}.pdf"
        loai = linh_kien[i % len(linh_kien)]
        ma_sp = f"{loai[:3].upper()}-{1000+i}"

        # Tạo file PDF
        c = canvas.Canvas(ten_file, pagesize=A4)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 780, f"DATASHEET: {ma_sp}")

        c.setFont("Helvetica", 12)
        noi_dung = [
            f"Component: {loai}",
            f"Part Number: {ma_sp}",
            f"Voltage: {round(3.3 + i * 0.5, 1)}V",
            f"Current: {100 + i * 10}mA",
            f"Temperature: -40 to +85 Celsius",
            f"Package: SOT-23",
            f"Manufacturer: Texas Instruments",
            f"Description: High performance {loai} for industrial use.",
            f"Applications: Power supply, signal processing, automation.",
            f"Storage: Keep in dry place below 30 Celsius.",
        ]

        y = 740
        for dong in noi_dung:
            c.drawString(50, y, dong)
            y -= 25

        c.save()
        print(f"  → Tạo xong: {ten_file}")

    print(f"[OK] Tạo xong {so_luong} file PDF tại '{thu_muc}/'")


# ==============================================================
# PHẦN 3: ĐỌC 1 FILE PDF (hàm nhỏ, dễ hiểu)
# ==============================================================

def doc_mot_file_pdf(duong_dan_file):
    """
    Đọc 1 file PDF và trả về danh sách các trang (pages).

    PyPDFLoader tự động:
    - Mở file PDF
    - Đọc từng trang
    - Trả về list các Document object (mỗi object = 1 trang)
    """
    try:
        # Tạo loader cho file này
        loader = PyPDFLoader(duong_dan_file)

        # Đọc tất cả các trang trong file
        # pages = [Document(page_content="nội dung trang 1", metadata={...}),
        #          Document(page_content="nội dung trang 2", metadata={...}), ...]
        pages = loader.load()

        return pages

    except Exception as loi:
        # Nếu file bị hỏng hoặc không đọc được → bỏ qua, không crash
        print(f"  [!] Không đọc được file: {duong_dan_file} — Lỗi: {loi}")
        return []  # Trả về danh sách rỗng


# ==============================================================
# PHẦN 4: CẮT NHỎ VĂN BẢN (Chunking)
# ==============================================================

def cat_thanh_chunks(danh_sach_pages):
    """
    Nhận vào: danh sách pages từ PDF
    Trả về:   danh sách chunks (đoạn văn bản nhỏ hơn)

    Tại sao phải cắt nhỏ?
    → AI có giới hạn "context window" — không đọc được file quá dài
    → Cắt nhỏ giúp tìm đúng đoạn liên quan khi query
    """

    # RecursiveCharacterTextSplitter: cắt text theo thứ tự ưu tiên:
    # 1. Cắt tại dấu xuống dòng đôi (\n\n) — ranh giới đoạn văn
    # 2. Nếu không đủ → cắt tại dấu xuống dòng (\n)
    # 3. Nếu không đủ → cắt tại dấu chấm (.)
    # 4. Nếu không đủ → cắt tại dấu cách ( )
    # → Cách này giữ nguyên ý nghĩa tốt hơn cắt cứng theo số ký tự
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],  # Thứ tự ưu tiên cắt
    )

    # Cắt toàn bộ pages thành chunks
    chunks = splitter.split_documents(danh_sach_pages)

    return chunks


# ==============================================================
# PHẦN 5: VÒNG LẶP CHÍNH — XỬ LÝ THEO BATCH
# ==============================================================

def xu_ly_tat_ca_pdf(thu_muc, batch_size=50):
    """
    Đọc và chunk toàn bộ PDF trong thư mục,
    xử lý theo từng batch để bảo vệ RAM.

    Tại sao dùng batch?
    → Nếu load 1000 file cùng lúc vào RAM → máy có thể crash
    → Batch 50 file = xử lý xong → giải phóng RAM → xử lý tiếp
    """

    # Lấy danh sách tất cả file PDF trong thư mục
    tat_ca_file = sorted(glob.glob(f"{thu_muc}/*.pdf"))
    tong_so_file = len(tat_ca_file)

    if tong_so_file == 0:
        print(f"[!] Không tìm thấy file PDF nào trong '{thu_muc}/'")
        print(f"    Hãy chép file PDF vào thư mục đó rồi chạy lại.")
        return []

    print(f"\n{'='*55}")
    print(f"  BƯỚC 1: ĐỌC & CHUNK TÀI LIỆU PDF")
    print(f"{'='*55}")
    print(f"[INFO] Tìm thấy {tong_so_file} file PDF")
    print(f"[INFO] Batch size: {batch_size} file/lần")
    print(f"[INFO] Chunk size: {CHUNK_SIZE} ký tự")
    print(f"[INFO] Chunk overlap: {CHUNK_OVERLAP} ký tự")
    print(f"{'='*55}\n")

    # Danh sách lưu TẤT CẢ chunks từ mọi file
    tat_ca_chunks = []

    # --- VÒNG LẶP BATCH ---
    # range(0, 100, 50) → [0, 50]  (batch 1: file 0-49, batch 2: file 50-99)
    for bat_dau in range(0, tong_so_file, batch_size):

        # Lấy slice của danh sách file cho batch này
        # VD: batch_size=50, bat_dau=0 → batch_files = file[0:50]
        #                    bat_dau=50 → batch_files = file[50:100]
        ket_thuc = min(bat_dau + batch_size, tong_so_file)
        batch_files = tat_ca_file[bat_dau:ket_thuc]

        so_batch = (bat_dau // batch_size) + 1
        tong_batch = (tong_so_file + batch_size - 1) // batch_size
        print(f"[BATCH {so_batch}/{tong_batch}] Đang xử lý file {bat_dau+1} → {ket_thuc}...")

        chunks_cua_batch = []

        # --- VÒNG LẶP FILE (bên trong batch) ---
        for i, duong_dan in enumerate(batch_files):
            chi_so_toan_cuc = bat_dau + i + 1  # Số thứ tự tổng (1, 2, 3, ...)
            ten_file = os.path.basename(duong_dan)  # Chỉ lấy tên file, bỏ đường dẫn

            print(f"  [INFO] Đang xử lý datasheet {chi_so_toan_cuc}/{tong_so_file} — {ten_file}")

            # 1. Đọc file PDF
            pages = doc_mot_file_pdf(duong_dan)
            if not pages:
                continue  # File lỗi → bỏ qua, xử lý file tiếp theo

            # 2. Cắt thành chunks
            chunks = cat_thanh_chunks(pages)

            # 3. Thêm vào danh sách của batch
            chunks_cua_batch.extend(chunks)

            # In thông tin chi tiết về file vừa xử lý
            print(f"         → {len(pages)} trang | {len(chunks)} chunks")

        # Sau khi xử lý xong 1 batch → gộp vào danh sách tổng
        tat_ca_chunks.extend(chunks_cua_batch)

        print(f"  ✓ Batch xong! Chunks trong batch: {len(chunks_cua_batch)}")
        print(f"  ✓ Tổng chunks tích lũy: {len(tat_ca_chunks)}\n")

    return tat_ca_chunks


# ==============================================================
# PHẦN 6: IN KẾT QUẢ ĐỂ KIỂM TRA
# ==============================================================

def in_ket_qua(tat_ca_chunks):
    """In thông tin tổng kết và xem thử nội dung chunk."""

    print(f"\n{'='*55}")
    print(f"  KẾT QUẢ BƯỚC 1")
    print(f"{'='*55}")
    print(f"[OK] Tổng số chunks tạo được: {len(tat_ca_chunks)}")

    if not tat_ca_chunks:
        return

    # Xem thử 3 chunks đầu tiên
    print(f"\n--- XEM THỬ 3 CHUNK ĐẦU TIÊN ---")
    for i, chunk in enumerate(tat_ca_chunks[:3]):
        print(f"\n[CHUNK #{i+1}]")
        print(f"  Nguồn file : {chunk.metadata.get('source', 'N/A')}")
        print(f"  Trang số   : {chunk.metadata.get('page', 'N/A')}")
        print(f"  Độ dài     : {len(chunk.page_content)} ký tự")
        print(f"  Nội dung   :")
        print(f"  {'-'*45}")
        # In tối đa 300 ký tự để dễ đọc
        print(f"  {chunk.page_content[:300]}")
        if len(chunk.page_content) > 300:
            print(f"  ... (còn {len(chunk.page_content)-300} ký tự nữa)")
        print(f"  {'-'*45}")

    print(f"\n[DONE] Bước 1 hoàn thành!")
    print(f"       Chunks đã sẵn sàng để đưa vào Vector Database (Bước 2)")


# ==============================================================
# CHẠY CHƯƠNG TRÌNH
# ==============================================================

def xu_ly_tat_ca_tai_lieu(thu_muc_pdf, thu_muc_document, batch_size=50):
    """
    Đọc và chunk toàn bộ tài liệu từ 3 nguồn:
      1. PDF  (nếu có) trong thu_muc_pdf
      2. Excel (.xlsx) trong thu_muc_document
      3. CSV  (.csv)   trong thu_muc_document
    Trả về danh sách chunk thống nhất để đưa vào Vector DB.
    """
    print(f"\n{'='*55}")
    print(f"  BƯỚC 1: ĐỌC & CHUNK TÀI LIỆU (PDF + Excel + CSV)")
    print(f"{'='*55}")

    tat_ca_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    # ── 1. PDF ──
    pdf_files = sorted(glob.glob(f"{thu_muc_pdf}/*.pdf"))
    if pdf_files:
        print(f"\n[PDF] Tìm thấy {len(pdf_files)} file")
        pdf_chunks = xu_ly_tat_ca_pdf(thu_muc_pdf, batch_size)
        tat_ca_chunks.extend(pdf_chunks)
    else:
        print(f"\n[PDF] Không có file PDF trong '{thu_muc_pdf}/'")

    # ── 2. Excel ──
    print(f"\n[EXCEL] Đọc từ '{thu_muc_document}/'")
    excel_docs = excel_to_documents(thu_muc_document)
    if excel_docs:
        excel_chunks = splitter.split_documents(excel_docs)
        print(f"[EXCEL] {len(excel_docs)} rows → {len(excel_chunks)} chunks")
        tat_ca_chunks.extend(excel_chunks)

    # ── 3. CSV ──
    print(f"\n[CSV] Đọc từ '{thu_muc_document}/'")
    csv_docs = csv_to_documents(thu_muc_document)
    if csv_docs:
        csv_chunks = splitter.split_documents(csv_docs)
        print(f"[CSV] {len(csv_docs)} rows → {len(csv_chunks)} chunks")
        tat_ca_chunks.extend(csv_chunks)

    print(f"\n{'='*55}")
    print(f"[TỔNG] {len(tat_ca_chunks)} chunks sẵn sàng cho Vector DB")
    print(f"{'='*55}")
    return tat_ca_chunks


if __name__ == "__main__":

    # Chạy pipeline đọc & chunk toàn bộ tài liệu
    tat_ca_chunks = xu_ly_tat_ca_tai_lieu(
        thu_muc_pdf=THU_MUC_PDF,
        thu_muc_document=THU_MUC_DOCUMENT,
        batch_size=BATCH_SIZE,
    )

    # In kết quả
    in_ket_qua(tat_ca_chunks)
