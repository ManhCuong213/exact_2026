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
    retrieve_context = lambda *_, **__: ""


# ============================================================
# CẤU HÌNH
# ============================================================
CONFIG = {
    "logic_file":   "data/Logic_Based_Educational_Queries.json",
    "physics_file": "data/Physics_Problems_Text_Only.csv",
    "output_file":  "results/output.json",
    "max_logic":    None,   # Số câu logic chạy thử (None = tất cả)
    "max_physics":  None,   # Số câu physics chạy thử (None = tất cả)
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
def run_logic_pipeline(logic_data: list, all_results: list, already_done: set) -> list:
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
            q_id = f"L{entry_idx+1}-Q{q_idx+1}"

            if q_id in already_done:
                continue

            total += 1
            log(f"\n[{q_id}] {pq['type'].upper()}: {pq['question'][:60]}...")

            try:
                context = retrieve_context(pq["question"], question_type="logic")
                solution = solve_logic(pq, context=context)

                explanation = generate_explanation(
                    question=pq["question"],
                    q_type=pq["type"],
                    solution=solution,
                    premises=pq["premises"]
                )

                is_correct = check_answer(solution["answer"], pq["expected_answer"])
                if is_correct:
                    correct += 1

                status = "✅" if is_correct else "❌"
                log(f"  AI: {solution['answer']} | Expected: {pq['expected_answer']} {status}")

                all_results.append({
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
                all_results.append({
                    "id": q_id,
                    "domain": "logic",
                    "answer": "Unknown",
                    "expected": pq["expected_answer"],
                    "correct": False,
                    "explanation": f"Error: {str(e)}"
                })

            if len(all_results) % 50 == 0:
                save_checkpoint(all_results)
                log(f"💾 Checkpoint saved: {len(all_results)} questions done")

    accuracy = correct / total * 100 if total > 0 else 0
    log(f"\n📊 Logic Accuracy: {correct}/{total} = {accuracy:.1f}%")
    return all_results


# ============================================================
# XỬ LÝ PHYSICS DATASET
# ============================================================
def run_physics_pipeline(physics_data: list, all_results: list, already_done: set) -> list:
    limit = CONFIG["max_physics"]
    data = physics_data[:limit] if limit else physics_data

    log(f"\n{'='*55}")
    log(f"⚡ PHYSICS PIPELINE — {len(data)} bài")
    log(f"{'='*55}")

    for row in data:
        pq = process_physics_entry(row)

        if pq["id"] in already_done:
            continue

        log(f"\n[{pq['id']}] {pq['question'][:60]}...")

        try:
            context = retrieve_context(pq["question"], question_type="physics")
            solution = solve_physics(pq, context=context)

            explanation = generate_explanation(
                question=pq["question"],
                q_type="physics",
                solution=solution
            )

            expected = pq["expected_answer"]
            ai_ans = str(solution["answer"])
            is_close = compare_physics_answer(
                f"{ai_ans} {solution['unit']}",
                f"{expected} {pq['unit']}"
            )

            status = "✅" if is_close else "❌"
            log(f"  AI: {ai_ans} {solution['unit']} | Expected: {expected} {pq['unit']} {status}")

            all_results.append({
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
            all_results.append({
                "id": pq["id"],
                "domain": "physics",
                "answer": "Error",
                "expected": pq["expected_answer"],
                "explanation": f"Error: {str(e)}"
            })

        if len(all_results) % 50 == 0:
            save_checkpoint(all_results)
            log(f"💾 Checkpoint saved: {len(all_results)} questions done")

    return all_results


# ============================================================
# CHECKPOINT
# ============================================================
CHECKPOINT_FILE = "results/checkpoint.json"


def load_checkpoint() -> list:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_checkpoint(results: list):
    os.makedirs("results", exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Delete checkpoint and start over")
    args = parser.parse_args()

    if args.fresh and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log("🗑️  Checkpoint deleted")

    all_results = load_checkpoint()
    already_done = {r["id"] for r in all_results}

    start_time = time.time()
    log("🚀 EXACT 2026 — Bắt đầu chạy hệ thống")
    log(f"{'='*55}")
    if CONFIG["max_logic"] is None:
        log("⚠️  Chạy TOÀN BỘ dataset — ước tính 2-3 tiếng")
    else:
        log(f"🧪 Chế độ test — chỉ chạy {CONFIG['max_logic']} logic + {CONFIG['max_physics']} physics")

    if all_results:
        log(f"▶ Resuming from question {len(all_results)}/6433")
    else:
        log("▶ Starting fresh run")

    # ── Chạy Logic ──
    try:
        with open(CONFIG["logic_file"], encoding="utf-8") as f:
            logic_data = json.load(f)
        log(f"✅ Đọc Logic dataset: {len(logic_data)} entries")
        run_logic_pipeline(logic_data, all_results, already_done)
    except FileNotFoundError:
        log(f"⚠️ Không tìm thấy {CONFIG['logic_file']}")

    # ── Chạy Physics ──
    try:
        with open(CONFIG["physics_file"], encoding="utf-8") as f:
            physics_data = list(csv.DictReader(f))
        log(f"✅ Đọc Physics dataset: {len(physics_data)} bài")
        run_physics_pipeline(physics_data, all_results, already_done)
    except FileNotFoundError:
        log(f"⚠️ Không tìm thấy {CONFIG['physics_file']}")

    # ── Lưu kết quả cuối ──
    save_checkpoint(all_results)
    os.makedirs("results", exist_ok=True)
    os.replace(CHECKPOINT_FILE, CONFIG["output_file"])

    # ── Báo cáo tổng kết ──
    elapsed = time.time() - start_time

    logic_correct = sum(1 for r in all_results if r.get("domain") == "logic" and r.get("correct"))
    logic_total   = sum(1 for r in all_results if r.get("domain") == "logic")

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
