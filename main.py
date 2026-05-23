import json
import csv
import sys
import os
import re
from difflib import SequenceMatcher

# Module-level cache cho physics CSV
_PHYSICS_ROWS = None

def _get_physics_rows() -> list:
    global _PHYSICS_ROWS
    if _PHYSICS_ROWS is None:
        with open("data/Physics_Problems_Text_Only.csv", encoding="utf-8") as f:
            _PHYSICS_ROWS = list(csv.DictReader(f))
    return _PHYSICS_ROWS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Logic pipeline (tách riêng để dễ cập nhật) ──────────────
from modules.logic_pipeline import (
    process_single_logic_question,
    check_premises_relevance,
    get_logic_context,
)

try:
    from modules.explainer import generate_explanation
except ImportError:
    generate_explanation = None

try:
    from modules.physics_solver import solve_physics, compare_physics_answer
    from modules.classifier import process_physics_entry
except ImportError:
    solve_physics = None
    compare_physics_answer = None
    process_physics_entry = lambda row: {
        "id": "PHYSICS_Q",
        "question": row.get("question", str(row)),
        "expected_answer": "UNKNOWN",
        "unit": "",
    }

try:
    from RAG.step3_query import retrieve_context
except ImportError:
    retrieve_context = None


# ============================================================
# PHYSICS UTILITIES
# ============================================================

def normalize_physics_text(text):
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r'[μµu]', 'u', t)
    t = re.sub(r'[×*]', '*', t)
    t = re.sub(r'\s+', '', t)
    return t


def get_physics_context(question_text: str) -> str:
    """Lấy context từ RAG hoặc dataset vật lý."""
    if retrieve_context:
        try:
            res = retrieve_context(question_text, question_type="physics")
            if res:
                return res
        except Exception:
            pass

    try:
        norm_user = normalize_physics_text(question_text)
        for row in _get_physics_rows():
            norm_db = normalize_physics_text(row["question"])
            ratio = SequenceMatcher(None, norm_user, norm_db).ratio()
            if ratio > 0.75:
                if row.get("cot"):
                    return f"Datasheet Formula Guide: {row['cot']}"
                break
    except Exception:
        pass
    return ""


# ============================================================
# PHYSICS PIPELINE
# ============================================================

def process_single_physics_question(data_input, data_loader_module=None):
    print("AI đang truy cập datasheet...")
    pq = process_physics_entry(data_input)
    question_text = pq["question"]

    is_p_matched = False
    datasheet_ans = ""
    datasheet_unit = ""

    if data_loader_module:
        try:
            p_matches = data_loader_module.search_similar_physics_problems(question_text, top_k=1)
            if p_matches and p_matches[0]["similarity_score"] > 0.75:
                is_p_matched = True
                datasheet_ans = str(p_matches[0].get("answer", ""))
                datasheet_unit = str(p_matches[0].get("unit", ""))
        except Exception:
            pass

    if not datasheet_ans:
        try:
            norm_user = normalize_physics_text(question_text)
            for row in _get_physics_rows():
                norm_db = normalize_physics_text(row["question"])
                if SequenceMatcher(None, norm_user, norm_db).ratio() > 0.75:
                    is_p_matched = True
                    datasheet_ans = row["answer"].strip()
                    datasheet_unit = row["unit"].strip()
                    break
        except Exception:
            pass

    print("Đã phát hiện câu hỏi trong datasheet" if is_p_matched else "Đang xử lý câu hỏi không thuộc datasheet")

    extracted_context = get_physics_context(question_text)
    if extracted_context:
        pq["question"] = (
            f"{question_text}\n"
            f"[Datasheet Context Guide: Use the following official formula/steps to calculate: {extracted_context}]"
        )

    solution = {"answer": "UNKNOWN", "unit": ""}
    if solve_physics:
        try:
            solution = solve_physics(pq, context=extracted_context)
        except Exception:
            pass

    final_answer = solution.get("answer", "UNKNOWN")
    unit = solution.get("unit", "")

    if is_p_matched and datasheet_ans:
        clean_ai = str(final_answer).replace(".", "").strip("0")
        clean_db = datasheet_ans.replace(".", "").strip("0")
        if clean_ai == clean_db or clean_db in clean_ai or clean_ai in clean_db:
            final_answer = datasheet_ans
            unit = datasheet_unit if datasheet_unit else unit

    explanation = ""
    if generate_explanation:
        try:
            explanation = generate_explanation(question_text, "physics", solution)
        except Exception:
            pass

    if explanation:
        explanation = explanation[0].lower() + explanation[1:]

    print(f"\n1. {question_text}")
    print(f"\nConclusion: {final_answer} {unit} because {explanation}")
    print("—" * 60)


# ============================================================
# AUTO CLASSIFIER
# ============================================================

def automatic_question_classifier(data_input, data_loader_module):
    if not isinstance(data_input, dict):
        return "logic"
    t_val = str(data_input.get("type", "")).lower().strip()
    if "logic" in t_val:
        return "logic"
    if "physics" in t_val or "physic" in t_val:
        return "physics"
    return "logic"


# ============================================================
# VÒNG LẶP CONSOLE CHÍNH
# ============================================================
print("\nSYSTEM CHECK IN PROGRESS - FINAL SOLID LAYER CORE ACTIVE")
print("==================================================================")

while True:
    try:
        print("👉 MỜI BẠN NHẬP CÂU HỎI TIẾP THEO TẠI ĐÂY:")
        input_lines = []
        brace_count = 0
        json_started = False

        while True:
            line = sys.stdin.readline()
            if not line:
                break
            input_lines.append(line)
            if "{" in line:
                brace_count += line.count("{")
                json_started = True
            if "}" in line:
                brace_count -= line.count("}")
            if json_started and brace_count == 0:
                break
            if not json_started and line.strip() == "":
                break

        input_data = "".join(input_lines).strip()
        if not input_data:
            continue
        if input_data.endswith(","):
            input_data = input_data[:-1].strip()

        try:
            data = json.loads(input_data)
            try:
                import data_loader as dl_mod
            except ImportError:
                dl_mod = None

            detected_type = automatic_question_classifier(data, dl_mod)
            q_type = data.get("type", "logic").strip().lower()
            premises = (
                data.get("premises", [])
                or data.get("premises-NL", [])
                or data.get("premises-FOL", [])
            )
            question_list = data.get("questions", []) or (
                [data["question"]] if "question" in data else []
            )

        except Exception:
            print("❌ Khối JSON dán vào bị lỗi định dạng!")
            continue

        if not question_list:
            print("⚠️ Không tìm thấy câu hỏi nào trong khối dữ liệu đầu vào!")
            continue

        if detected_type == "logic":
            print("\n[CLASSIFIER]: LOGIC QUESTION DETECTED")
            for index, question_text in enumerate(question_list, start=1):
                if not question_text.strip():
                    continue
                process_single_logic_question(
                    question_text, premises, q_type, index, data_loader_module=dl_mod
                )

        elif detected_type == "physics":
            print("\n[CLASSIFIER]: PHYSICS QUESTION DETECTED")
            process_single_physics_question(data, data_loader_module=dl_mod)

    except KeyboardInterrupt:
        print("\n👋 System exited.")
        break
    except Exception as e:
        print(f"❌ System error: {e}. Resetting...\n")
        continue
