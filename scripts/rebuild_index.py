"""
scripts/rebuild_index.py — Build FAISS vector index from Logic + Physics datasets.
Run from project root: python scripts/rebuild_index.py
"""

import json
import csv
import os
import sys

# Resolve project root regardless of where the script is called from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # older langchain versions

LOGIC_FILE   = os.path.join(ROOT, "data", "Logic_Based_Educational_Queries.json")
PHYSICS_FILE = os.path.join(ROOT, "data", "Physics_Problems_Text_Only.csv")
INDEX_PATH   = os.path.join(ROOT, "RAG", "faiss_index")
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


def build_logic_chunks() -> list:
    with open(LOGIC_FILE, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for i, entry in enumerate(data):
        premises_text = "\n".join(f"- {p}" for p in entry["premises-NL"])
        for question, answer, explanation in zip(
            entry["questions"], entry["answers"], entry["explanation"]
        ):
            text = (
                f"Question: {question}\n"
                f"Premises: {premises_text}\n"
                f"Answer: {answer}\n"
                f"Explanation: {explanation}"
            )
            docs.append(Document(
                page_content=text,
                metadata={"source": "logic", "type": "logic", "chunk_id": f"L{i}"}
            ))
    return docs


def build_physics_chunks() -> list:
    docs = []
    with open(PHYSICS_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("answer", "").strip():
                continue
            text = (
                f"Problem: {row['question']}\n"
                f"Solution method: {row['cot']}"
            )
            docs.append(Document(
                page_content=text,
                metadata={"source": "physics", "type": "physics"}
            ))
    return docs


def main():
    print("Building FAISS index from Logic + Physics datasets...")
    print()

    logic_docs   = build_logic_chunks()
    physics_docs = build_physics_chunks()
    all_docs     = logic_docs + physics_docs

    print(f"  Logic chunks  : {len(logic_docs)}")
    print(f"  Physics chunks: {len(physics_docs)}")
    print(f"  Total         : {len(all_docs)}")
    print()
    print(f"Loading embedding model: {EMBED_MODEL}")
    print("(First run downloads ~90 MB — subsequent runs use local cache)")
    print()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding all chunks and building FAISS index...")
    vector_db = FAISS.from_documents(all_docs, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    vector_db.save_local(INDEX_PATH)

    print()
    print(f"Index saved to: {INDEX_PATH}")
    print(f"  Logic chunks  : {len(logic_docs)}")
    print(f"  Physics chunks: {len(physics_docs)}")
    print(f"  Total vectors : {vector_db.index.ntotal}")


if __name__ == "__main__":
    main()
