"""
Test tìm kiếm thủ công — gõ câu hỏi bất kỳ, xem Vector DB trả về gì
Chạy: py test_search.py
"""
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load Vector DB đã tạo ở Bước 2
print("[INFO] Đang load Vector DB...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vector_db = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)
print(f"[OK] Load xong! Vectors trong DB: {vector_db.index.ntotal}")
print("="*50)
print("Gõ câu hỏi bất kỳ, Enter để tìm. Gõ 'exit' để thoát.")
print("="*50)

while True:
    cau_hoi = input("\nCâu hỏi của bạn: ").strip()
    if cau_hoi.lower() == "exit":
        break
    if not cau_hoi:
        continue

    ket_qua = vector_db.similarity_search(cau_hoi, k=3)
    print(f"\nTìm thấy {len(ket_qua)} kết quả liên quan:")
    for i, doc in enumerate(ket_qua):
        ten_file = os.path.basename(doc.metadata.get("source", "N/A"))
        print(f"\n  #{i+1} [{ten_file}]")
        print(f"  {doc.page_content[:200]}...")
