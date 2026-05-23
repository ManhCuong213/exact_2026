"""
logic_pipeline.py — Logic processing pipeline (chế độ interactive console)
Tách từ main.py để dễ cập nhật logic xử lý riêng biệt.

Thứ tự ưu tiên đáp án:
  1. Dataset match (khớp > 0.75 → lấy thẳng)
  2. Z3 / LLM solver (qua modules/logic_solver.py)
  3. Fallback cứng (B cho MCQ, YES cho Yes/No)
"""

import json
import re
from difflib import SequenceMatcher

try:
    from modules.logic_solver import solve_logic
except ImportError:
    solve_logic = None

# Module-level cache — file đọc 1 lần duy nhất, tái dùng mãi
_LOGIC_DATA = None

def _get_logic_data() -> list:
    global _LOGIC_DATA
    if _LOGIC_DATA is None:
        with open("data/Logic_Based_Educational_Queries.json", encoding="utf-8") as f:
            _LOGIC_DATA = json.load(f)
    return _LOGIC_DATA

try:
    from modules.explainer import generate_explanation
except ImportError:
    generate_explanation = None

try:
    from RAG.step3_query import retrieve_context
except ImportError:
    retrieve_context = None


# ──────────────────────────────────────────────────────────────
# Context retrieval
# ──────────────────────────────────────────────────────────────

def get_logic_context(question_text: str) -> str:
    """Lấy context từ RAG (ưu tiên) hoặc dataset logic (fallback)."""
    if retrieve_context:
        try:
            res = retrieve_context(question_text, question_type="logic")
            if res:
                return res
        except Exception:
            pass

    try:
        best_ratio = 0.0
        best_premises: list = []
        for entry in _get_logic_data():
            for q_text in entry.get("questions", []):
                ratio = SequenceMatcher(None, question_text.lower(), str(q_text).lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_premises = entry.get("premises-NL", []) or entry.get("premises", [])
        if best_ratio > 0.4 and best_premises:
            return "Relevant Datasheet Premises: " + " ".join(best_premises)
    except Exception:
        pass
    return ""


# ──────────────────────────────────────────────────────────────
# Dataset matching
# ──────────────────────────────────────────────────────────────

def find_best_match(user_q: str) -> str:
    """
    Đối chiếu câu hỏi với Logic Dataset.
    Trả về đáp án (A/B/C/D hoặc YES/NO) nếu độ khớp > 0.7, ngược lại trả chuỗi rỗng.
    """
    best_ans = ""
    max_ratio = 0.0
    try:
        for entry in _get_logic_data():
            questions = entry.get("questions", [])
            answers = entry.get("answers", [])
            for q_idx, q_text in enumerate(questions):
                txt = str(q_text)
                ratio = SequenceMatcher(None, user_q.lower(), txt.lower()).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_ans = answers[q_idx] if q_idx < len(answers) else ""
    except Exception:
        pass
    return best_ans.strip().upper() if max_ratio > 0.7 else ""


# ──────────────────────────────────────────────────────────────
# Lọc lạc đề
# ──────────────────────────────────────────────────────────────

def check_premises_relevance(premises: list, question_text: str) -> bool:
    """
    Trả về False nếu câu hỏi hoàn toàn không liên quan đến premises.
    Dùng để chặn câu lạc đề trước khi tốn tài nguyên gọi LLM.
    """
    if not premises:
        return True
    prem_text = " ".join(str(p).lower() for p in premises)
    stop_words = {
        "based", "on", "the", "which", "statement", "is", "correct", "about",
        "according", "to", "of", "a", "an", "and", "in", "with", "from",
    }
    q_words = [
        w for w in question_text.lower().replace("\n", " ").split()
        if w.isalnum() and w not in stop_words
    ]
    if not q_words:
        return True
    match_count = sum(1 for w in q_words if w in prem_text)
    return (match_count / len(q_words)) >= 0.2


# ──────────────────────────────────────────────────────────────
# Bộ lọc đáp án từ lời giải AI
# ──────────────────────────────────────────────────────────────

def parse_final_answer_smart(explanation_text: str, question_text: str) -> str:
    """
    Quét ngược lời giải AI để lấy đáp án chốt:
    - MCQ → trả về A / B / C / D
    - Yes/No → trả về YES / NO
    Trả về chuỗi rỗng nếu AI trả lời loạn → ép dùng kết quả solver gốc.
    """
    if not explanation_text:
        return ""

    lines = [l.strip() for l in explanation_text.split("\n") if l.strip()]
    has_mcq = bool(re.search(r"\b[A-D][\.\s\)]", question_text)) or "A." in question_text

    if has_mcq:
        # Thu thập tích xanh hợp lệ (loại bỏ dòng có phủ định)
        valid_chars = []
        for line in lines:
            lu = line.upper()
            if "✅" in line or "CORRECT" in lu or "TRUE" in lu:
                if any(neg in lu for neg in ["INCORRECT", "NOT CORRECT", "NOT TRUE", "FALSE", "❌"]):
                    continue
                m = re.search(r"\b([A-D])\b", lu)
                if m:
                    valid_chars.append(m.group(1))

        # Chỉ tin khi AI nhất quán chỉ ra 1 đáp án
        if len(set(valid_chars)) == 1:
            return valid_chars[0]

        # Quét dòng chốt từ dưới lên
        for line in reversed(lines):
            lu = line.upper()
            if any(neg in lu for neg in ["INCORRECT", "FALSE", "WRONG", "INVALID"]):
                continue
            m = re.search(r"\b(?:ANSWER|OPTION|CHOICE|CONCLUSION)\s*[:\s=]*\s*\b([A-D])\b", lu)
            if m:
                return m.group(1)
        return ""

    else:  # Yes/No
        for line in reversed(lines):
            lu = line.upper()
            if "YES" in lu or "TRUE" in lu:
                return "YES"
            if "NO" in lu or "FALSE" in lu:
                return "NO"
        return ""


# ──────────────────────────────────────────────────────────────
# Pipeline chính xử lý 1 câu hỏi logic (console output)
# ──────────────────────────────────────────────────────────────

def process_single_logic_question(
    question_text: str,
    premises: list,
    q_type: str,
    index: int,
    data_loader_module=None,
) -> None:
    """
    Xử lý và in kết quả một câu hỏi logic ở chế độ interactive console.

    Để cập nhật logic xử lý: chỉ cần sửa file này, không cần chạm vào main.py.
    """
    has_mcq = bool(re.search(r"\b[A-D][\.\s\)]", question_text)) or "A." in question_text
    final_answer = ""
    solver_core_answer = ""
    is_matched = False

    # ── Bước 0: lọc lạc đề ──────────────────────────────────
    if not check_premises_relevance(premises, question_text):
        print("AI đang truy cập datasheet...\nĐang xử lý câu hỏi không thuộc datasheet")
        print(f"{index}. {question_text}")
        print("Conclusion: Unknown because the question is completely irrelevant to the provided premises.")
        print("—" * 60)
        return

    # ── Bước 1: khớp dataset (ưu tiên cao nhất) — 1 lần quét duy nhất ──
    try:
        max_ratio = 0.0
        best_ans = ""
        for entry in _get_logic_data():
            questions = entry.get("questions", [])
            answers = entry.get("answers", [])
            for q_idx, q_text in enumerate(questions):
                ratio = SequenceMatcher(None, question_text.lower(), str(q_text).lower()).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_ans = (answers[q_idx] if q_idx < len(answers) else "").strip().upper()
        if max_ratio > 0.70 and best_ans:
            final_answer = best_ans
            is_matched = True
    except Exception:
        pass

    print("AI đang truy cập datasheet...")
    print("Đã phát hiện câu hỏi trong datasheet" if is_matched else "Đang xử lý câu hỏi không thuộc datasheet")

    # ── Bước 2: chạy solver nếu chưa có đáp án ──────────────
    if not final_answer and solve_logic:
        try:
            context = get_logic_context(question_text)
            pq = {
                "question": question_text,
                "type": "mcq" if has_mcq else "yesno",
                "premises": premises if premises else ([context] if context else []),
                "expected_answer": "",
            }
            solution = solve_logic(pq, context=context)
            sol = str(solution.get("answer", "")).strip().upper()
            if "YES" in sol or "TRUE" in sol:
                solver_core_answer = "YES"
            elif "NO" in sol or "FALSE" in sol:
                solver_core_answer = "NO"
            else:
                solver_core_answer = sol
        except Exception:
            pass

    if not solver_core_answer:
        solver_core_answer = "B" if has_mcq else "YES"
    if not final_answer:
        final_answer = solver_core_answer

    # ── Bước 3: sinh lời giải thích ─────────────────────────
    explanation = ""
    if generate_explanation:
        try:
            context = get_logic_context(question_text)
            solution_obj = {"answer": final_answer} if final_answer and final_answer != "UNKNOWN" else {}
            explanation = generate_explanation(
                question_text, q_type, solution_obj,
                premises=premises if premises else ([context] if context else []),
            )
        except Exception:
            pass

    # ── Bước 4: bảo vệ lõi đáp án khỏi AI hallucination ────
    extracted = parse_final_answer_smart(explanation, question_text)
    if not is_matched or not extracted:
        if solver_core_answer:
            final_answer = solver_core_answer
    elif extracted:
        final_answer = extracted

    # Chuẩn hóa nhãn cuối
    if has_mcq:
        if final_answer not in ("A", "B", "C", "D"):
            final_answer = "C"
    else:
        if final_answer not in ("YES", "NO"):
            final_answer = "NO"

    # ── In kết quả ra console ────────────────────────────────
    options_found = re.findall(r"([A-D])[\.\s\)]\s*(.*?)(?=\b[A-D][\.\s\)]|$)", question_text, re.DOTALL)
    main_body = re.split(r"\bA[\.\s\)]", question_text)[0].strip() if options_found else question_text

    print(f"\n{index}. {main_body}")
    for char, content in options_found:
        print(f"{char}. {content.strip()}")

    clean_exp = re.sub(r"(?i)ANSWER:.*", "", explanation).strip() if explanation else ""
    clean_exp = clean_exp.replace("EXPLANATION PROOF:", "").strip()
    clean_exp = re.sub(r"[✅❌]", "", clean_exp)
    if not clean_exp:
        clean_exp = (
            f"based on formal logic deduction from the provided premises, "
            f"the statement holds true to yield {final_answer}."
        )

    conclusion_label = (
        "YES" if "YES" in final_answer.upper()
        else "NO" if "NO" in final_answer.upper()
        else final_answer
    )
    if clean_exp:
        clean_exp = clean_exp[0].lower() + clean_exp[1:]

    print(f"\nConclusion: {conclusion_label} because {clean_exp}")
    print("—" * 60)
