"""
classifier.py — Phân loại và làm sạch câu hỏi
Người phụ trách: Member 2
"""

import re


def clean_text(text: str) -> str:
    """Làm sạch ký tự lỗi encoding trong dataset"""
    text = text.replace("?forall", "∀")
    text = text.replace("?exists", "∃")
    text = re.sub(r'[^\x00-\x7F∀∃¬→∧∨∩∪]+', '', text)
    return text.strip()


def classify_question_type(question: str) -> str:
    """
    Phân loại câu hỏi thành: 'yesno' hoặc 'mcq'
    """
    question_lower = question.lower()
    if any(opt in question_lower for opt in ["\na.", "\nb.", "\nc.", "\nd.", "option a", "option b"]):
        return "mcq"
    if question_lower.startswith("which") and ("a." in question_lower or "b." in question_lower):
        return "mcq"
    return "yesno"


def extract_options(question: str) -> dict:
    """
    Trích xuất các lựa chọn A, B, C, D từ câu hỏi MCQ
    """
    options = {}
    pattern = r'\n([A-D])\.\s*(.+?)(?=\n[A-D]\.|$)'
    matches = re.findall(pattern, question, re.DOTALL)
    for letter, content in matches:
        options[letter] = content.strip()
    return options


def get_relevant_premises(entry: dict, question_idx: int) -> list:
    """
    Dùng field 'idx' để lấy đúng premises liên quan đến câu hỏi
    Input:  1 entry từ JSON, index của câu hỏi (0 hoặc 1)
    Output: list premises liên quan
    """
    premises_nl = entry.get("premises-NL", [])
    idx = entry.get("idx", [])

    # Nếu có idx hợp lệ → lọc premises
    if idx and question_idx < len(idx):
        relevant_idx = idx[question_idx]
        if isinstance(relevant_idx, list) and len(relevant_idx) > 0:
            # idx dùng 1-based index
            filtered = []
            for i in relevant_idx:
                if isinstance(i, int) and 1 <= i <= len(premises_nl):
                    filtered.append(premises_nl[i - 1])
            if filtered:
                return filtered

    # Fallback: trả về tất cả premises
    return premises_nl


def process_logic_entry(entry: dict) -> list:
    """
    Xử lý 1 entry từ Logic dataset
    Input:  1 entry JSON gốc
    Output: list các câu hỏi đã được xử lý sạch
    """
    questions = entry.get("questions", [])
    answers = entry.get("answers", [])
    processed = []

    for i, (question, answer) in enumerate(zip(questions, answers)):
        q_type = classify_question_type(question)
        premises = get_relevant_premises(entry, i)
        premises = [clean_text(p) for p in premises]
        options = extract_options(question) if q_type == "mcq" else {}

        processed.append({
            "type": q_type,
            "question": clean_text(question),
            "premises": premises,
            "options": options,
            "expected_answer": answer  # Dùng để chấm điểm
        })

    return processed


def process_physics_entry(row: dict) -> dict:
    """
    Xử lý 1 hàng từ Physics CSV dataset
    Input:  1 row từ CSV
    Output: dict chuẩn cho physics_solver
    """
    return {
        "id": row.get("id", ""),
        "type": "physics",
        "question": row.get("question", "").strip(),
        "expected_answer": row.get("answer", "").strip(),
        "unit": row.get("unit", "").strip(),
        "cot_hint": row.get("cot", "").strip()  # Chain-of-thought gợi ý
    }


# ── Test thử ──
if __name__ == "__main__":
    import json

    print("Test classifier.py...")

    # Test với entry mẫu
    sample = {
        "idx": [[1, 4], [1, 2, 4]],
        "premises-NL": [
            "If a student completes all courses, they graduate.",
            "If a student graduates, they get a diploma.",
            "All students study hard.",
            "John completed all courses."
        ],
        "questions": [
            "Does John graduate?",
            "Which is correct?\nA. John gets a diploma\nB. John fails\nC. John drops out\nD. Unknown"
        ],
        "answers": ["Yes", "A"]
    }

    results = process_logic_entry(sample)
    for i, r in enumerate(results):
        print(f"\nCâu {i+1}:")
        print(f"  Type    : {r['type']}")
        print(f"  Premises: {len(r['premises'])} câu")
        print(f"  Options : {r['options']}")
        print(f"  Expected: {r['expected_answer']}")

    print("\n✅ classifier.py hoạt động!")
