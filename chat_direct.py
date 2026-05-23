import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from difflib import SequenceMatcher
from data_loader import load_logic_data, load_physics_data
from modules.logic_solver import solve_logic
from modules.physics_solver import solve_physics
from modules.classifier import classify_question_type
from modules.explainer import generate_explanation


def find_best_match(user_q: str, q_type: str) -> str:
    """Tìm câu tương tự trong dataset, trả về chuỗi context nếu khớp > 0.4."""
    best_context = ""
    max_ratio = 0.0

    if q_type == "logic":
        for entry in load_logic_data():
            for q_text in entry.get("questions", []):
                if not isinstance(q_text, str):
                    continue
                ratio = SequenceMatcher(None, user_q.lower(), q_text.lower()).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    answers = entry.get("answers", [])
                    best_ans = answers[entry["questions"].index(q_text)] if answers else ""
                    best_context = f"Question: {q_text}\nAnswer: {best_ans}"
    else:
        physics_df = load_physics_data()
        for _, row in physics_df.iterrows():
            txt = str(row.get("question", ""))
            ratio = SequenceMatcher(None, user_q.lower(), txt.lower()).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
                best_context = (
                    f"Question: {txt}\n"
                    f"Formula: {row.get('cot', '')}\n"
                    f"Answer: {row.get('answer', '')} {row.get('unit', '')}"
                )

    return best_context if max_ratio > 0.4 else ""


print("HE THONG CHAT DIRECT EXACT 2026")
print("====================================================")

while True:
    question = input("\n Nhap cau hoi cua ban (hoac go 'exit' de thoat): ")
    if question.lower() == "exit":
        break

    q_type = input("Nhap loai cau hoi ('logic' hoac 'physics'): ").strip().lower()
    if q_type not in ("logic", "physics"):
        print("  Loai khong hop le! Vui long nhap lai.")
        continue

    print("\n Dang quet dataset tim cau tuong tu...")
    context = find_best_match(question, q_type)
    if context:
        print("  Tim thay ngu canh tuong dong trong dataset!")
    else:
        print("  Khong tim thay cau nao giong, AI se tu giai.")

    print("  AI dang lap luan va giai bai...")

    if q_type == "logic":
        q_detected = classify_question_type(question)
        pq = {
            "question": question,
            "type": q_detected,
            "premises": [],
            "options": {},
            "expected_answer": "",
        }
        solution = solve_logic(pq, context=context)
        explanation = generate_explanation(question, q_detected, solution, premises=[])
        print(f"\n DAP AN AI: {solution.get('answer', 'Khong ro')}")
    else:
        pq = {"id": "USER_Q", "question": question, "expected_answer": "", "unit": "", "cot_hint": ""}
        solution = solve_physics(pq, context=context)
        explanation = generate_explanation(question, "physics", solution)
        print(f"\n DAP AN AI: {solution.get('answer', 'Khong ro')} {solution.get('unit', '')}")

    print(f"  GIAI THICH CHI TIET:\n{explanation}")
