"""
main.py — Pipeline chính tích hợp toàn bộ hệ thống EXACT 2026
Người phụ trách: Bạn (PM)
"""

import json
import csv
import time
import os
import sys

# Import các module của nhóm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.classifier import process_logic_entry, process_physics_entry
from modules.logic_solver import solve_logic
from modules.physics_solver import solve_physics, compare_physics_answer
from modules.explainer import generate_explanation
try:
    from RAG.step3_query import retrieve_context
except ImportError:
    retrieve_context = lambda *args, **kwargs: ""


# ============================================================
# CẤU HÌNH
# ============================================================
CONFIG = {
    "logic_file":   "data/Logic_Based_Educational_Queries.json",
    "physics_file": "data/Physics_Problems_Text_Only.csv",
    "output_file":  "results/output.json",
    "max_logic":    10,   # Số câu logic chạy thử (None = tất cả)
    "max_physics":  10,   # Số câu physics chạy thử (None = tất cả)
    "show_progress": True
}


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def log(msg: str):
    if CONFIG["show_progress"]:
        print(msg)


def check_answer(ai_answer: str, expected: str) -> bool:
    """So sánh đáp án AI với đáp án đúng"""
    return ai_answer.strip().upper() == expected.strip().upper()


# ============================================================
# XỬ LÝ LOGIC DATASET
# ============================================================
def run_logic_pipeline(logic_data: list) -> list:
    results = []
    limit = CONFIG["max_logic"]
    data = logic_data[:limit] if limit else logic_data

    log(f"\n{'='*55}")
    log(f"🧠 LOGIC PIPELINE — {len(data)} entries")
    log(f"{'='*55}")

    correct = 0
    total = 0

    for entry_idx, entry in enumerate(data):
        processed_questions = process_logic_entry(entry)

        for q_idx, pq in enumerate(processed_questions):
            total += 1
            q_id = f"L{entry_idx+1}-Q{q_idx+1}"

            log(f"\n[{q_id}] {pq['type'].upper()}: {pq['question'][:60]}...")

            try:
                # Bước 1: Lấy context từ RAG + Giải logic
                context = retrieve_context(pq["question"], question_type="logic")
                solution = solve_logic(pq, context=context)

                # Bước 2: Sinh giải thích
                explanation = generate_explanation(
                    question=pq["question"],
                    q_type=pq["type"],
                    solution=solution,
                    premises=pq["premises"]
                )

                # Bước 3: Chấm điểm
                is_correct = check_answer(
                    solution["answer"],
                    pq["expected_answer"]
                )
                if is_correct:
                    correct += 1

                status = "✅" if is_correct else "❌"
                log(f"  AI: {solution['answer']} | Expected: {pq['expected_answer']} {status}")

                results.append({
                    "id": q_id,
                    "domain": "logic",
                    "type": pq["type"],
                    "question": pq["question"],
                    "answer": solution["answer"],
                    "expected": pq["expected_answer"],
                    "correct": is_correct,
                    "confidence": solution.get("confidence", 0),
                    "explanation": explanation
                })

            except Exception as e:
                log(f"  ⚠️ Lỗi: {e}")
                results.append({
                    "id": q_id,
                    "domain": "logic",
                    "answer": "Unknown",
                    "expected": pq["expected_answer"],
                    "correct": False,
                    "explanation": f"Error: {str(e)}"
                })

    accuracy = correct / total * 100 if total > 0 else 0
    log(f"\n📊 Logic Accuracy: {correct}/{total} = {accuracy:.1f}%")
    return results


# ============================================================
# XỬ LÝ PHYSICS DATASET
# ============================================================
def run_physics_pipeline(physics_data: list) -> list:
    results = []
    limit = CONFIG["max_physics"]
    data = physics_data[:limit] if limit else physics_data

    log(f"\n{'='*55}")
    log(f"⚡ PHYSICS PIPELINE — {len(data)} bài")
    log(f"{'='*55}")

    for row in data:
        pq = process_physics_entry(row)
        log(f"\n[{pq['id']}] {pq['question'][:60]}...")

        try:
            # Bước 1: Lấy context từ RAG + Giải vật lý
            context = retrieve_context(pq["question"], question_type="physics")
            solution = solve_physics(pq, context=context)

            # Bước 2: Sinh giải thích
            explanation = generate_explanation(
                question=pq["question"],
                q_type="physics",
                solution=solution
            )

            # Bước 3: Chấm điểm sơ bộ
            expected = pq["expected_answer"]
            ai_ans = str(solution["answer"])
            is_close = compare_physics_answer(
                f"{ai_ans} {solution['unit']}",
                f"{expected} {pq['unit']}"
            )

            status = "✅" if is_close else "❌"
            log(f"  AI: {ai_ans} {solution['unit']} | Expected: {expected} {pq['unit']} {status}")

            results.append({
                "id": pq["id"],
                "domain": "physics",
                "question": pq["question"],
                "answer": f"{solution['answer']} {solution['unit']}",
                "expected": f"{expected} {pq['unit']}",
                "correct": is_close,
                "formula": solution["formula"],
                "explanation": explanation
            })

        except Exception as e:
            log(f"  ⚠️ Lỗi: {e}")
            results.append({
                "id": pq["id"],
                "domain": "physics",
                "answer": "Error",
                "expected": pq["expected_answer"],
                "explanation": f"Error: {str(e)}"
            })

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()

    log("🚀 EXACT 2026 — Bắt đầu chạy hệ thống")
    log(f"{'='*55}")

    all_results = []

    # ── Chạy Logic ──
    try:
        from data_loader import load_logic_data
        logic_data = load_logic_data()
        log(f"✅ Đọc Logic dataset: {len(logic_data)} entries")
        logic_results = run_logic_pipeline(logic_data)
        all_results.extend(logic_results)
    except Exception as e:
        log(f"⚠️ Lỗi load logic data: {e}")

    # ── Chạy Physics ──
    try:
        from data_loader import load_physics_data
        physics_df = load_physics_data()
        physics_data = physics_df.to_dict("records")
        log(f"✅ Đọc Physics dataset: {len(physics_data)} bài")
        physics_results = run_physics_pipeline(physics_data)
        all_results.extend(physics_results)
    except Exception as e:
        log(f"⚠️ Lỗi load physics data: {e}")

    # ── Lưu kết quả ──
    os.makedirs("results", exist_ok=True)
    with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── Báo cáo tổng kết ──
    elapsed = time.time() - start_time

    logic_correct   = sum(1 for r in all_results if r.get("domain") == "logic"   and r.get("correct"))
    logic_total     = sum(1 for r in all_results if r.get("domain") == "logic")
    physics_correct = sum(1 for r in all_results if r.get("domain") == "physics" and r.get("correct"))
    physics_total   = sum(1 for r in all_results if r.get("domain") == "physics")
    overall_correct = logic_correct + physics_correct
    overall_total   = logic_total + physics_total

    def _pct(a, b):
        return f"{a/b*100:.1f}%" if b > 0 else "N/A"

    log(f"\n{'='*55}")
    log(f"🎉 HOÀN THÀNH!")
    log(f"{'='*55}")
    log(f"⏱️  Thời gian chạy   : {elapsed:.1f} giây")
    log(f"📝 Tổng câu hỏi     : {overall_total}")
    if logic_total > 0:
        log(f"🧠 Logic accuracy   : {logic_correct}/{logic_total} = {_pct(logic_correct, logic_total)}")
    if physics_total > 0:
        log(f"⚡ Physics accuracy : {physics_correct}/{physics_total} = {_pct(physics_correct, physics_total)}")
    if overall_total > 0:
        log(f"🎯 Overall accuracy : {overall_correct}/{overall_total} = {_pct(overall_correct, overall_total)}")
    log(f"📁 Kết quả lưu tại  : {CONFIG['output_file']}")
    log(f"{'='*55}")


if __name__ == "__main__":
    main()
