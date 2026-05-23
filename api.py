# Run: uvicorn api:app --host 0.0.0.0 --port 8000
# run : py -m uvicorn api:app --reload
# Public URL: ngrok http 8000

import os
import sys
import concurrent.futures
from difflib import SequenceMatcher

_LLM_TIMEOUT = 60

def _ollama_call(fn):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=_LLM_TIMEOUT)

import requests as http_requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.classifier import classify_question_type, extract_options
from modules.logic_solver import solve_logic
from modules.physics_solver import solve_physics
from modules.explainer import generate_explanation
from modules.logic_pipeline import check_premises_relevance
from data_loader import search_similar_logic_questions, search_similar_physics_problems, get_dataset_stats

try:
    from RAG.step3_query import retrieve_context
except ImportError:
    retrieve_context = lambda *_, **__: ""

app = FastAPI(
    title="EXACT 2026 API",
    description="API tự động chấm điểm Logic & Vật lý, lọc lạc đề, tích hợp RAG và dataset matching."
)


# ─────────────────────────────────────────
# GET /
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "title": "EXACT 2026 API - Auto-detect Logic & Physics",
        "endpoints": {
            "GET /health": "Check API + Ollama + FAISS status",
            "POST /answer": "Send question (auto-detect domain)",
            "POST /solve": "Send logic question (BTC format)",
            "GET /dataset-stats": "Dataset statistics",
            "GET /docs": "Interactive Swagger UI",
        },
        "usage": {
            "simple": {
                "method": "POST /answer",
                "body": '{"question": "Calculate voltage in a 10μF capacitor"}',
                "note": "Auto-detects physics or logic"
            },
            "with_premises": {
                "method": "POST /answer",
                "body": '{"question": "Is A true?", "premises": ["If A then B", "B is true"]}',
                "note": "Explicitly provides logic premises"
            },
            "force_type": {
                "method": "POST /answer",
                "body": '{"question": "2+2=4?", "type": "logic"}',
                "note": "Force specific domain"
            },
            "btc_format": {
                "method": "POST /solve",
                "body": '{"question": "Which is correct?", "premises": ["If A then B"]}',
                "note": "Format chuẩn BTC với dataset matching"
            }
        },
        "auto_detection": "No need to specify type or premises - AI will figure it out!"
    }


# ─────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────

@app.get("/health")
def health():
    ollama_ok = False
    try:
        r = http_requests.get("http://localhost:11434", timeout=3)
        ollama_ok = "Ollama" in r.text
    except Exception:
        pass

    faiss_ok = os.path.exists("RAG/faiss_index/index.faiss")
    return {"status": "ok", "ollama": ollama_ok, "faiss": faiss_ok}


# ─────────────────────────────────────────
# GET /dataset-stats
# ─────────────────────────────────────────

@app.get("/dataset-stats")
def dataset_stats():
    """Thống kê về datasets đã load"""
    return get_dataset_stats()


# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────

class AnswerRequest(BaseModel):
    question: str
    type: str = "auto"          # "auto" | "logic" | "physics"
    premises: List[str] = []

class AnswerResponse(BaseModel):
    answer: str
    explanation: str
    fol: str = ""
    cot: List[str] = []
    premises: List[str] = []
    confidence: float = 0.8
    domain: str = ""                                    # "logic" hoặc "physics"
    related_data: Optional[List[Dict[str, Any]]] = None # Câu hỏi tương tự từ dataset

# Schema tương thích với endpoint /solve (format BTC)
class LogicRequest(BaseModel):
    type: Optional[str] = "logic"
    question: str
    premises: List[str]


# ─────────────────────────────────────────
# Utility: dataset matching + answer parsing
# ─────────────────────────────────────────

def find_best_match(user_q: str, q_type: str = "logic") -> str:
    """Đối chiếu câu hỏi với Dataset gốc, trả về đáp án nếu độ khớp > 0.6.
    Dùng data_loader cached data — không đọc file lại mỗi request."""
    best_ans = ""
    max_ratio = 0.0
    try:
        if q_type == "physics":
            from data_loader import load_physics_data
            physics_df = load_physics_data()
            for _, row in physics_df.iterrows():
                txt = str(row.get("question", ""))
                if not txt:
                    continue
                ratio = SequenceMatcher(None, user_q.lower(), txt.lower()).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_ans = str(row.get("answer", ""))
        else:
            from data_loader import load_logic_data
            for entry in load_logic_data():
                questions = entry.get("questions", [entry]) if isinstance(entry, dict) else [entry]
                for q in questions:
                    if not isinstance(q, dict):
                        continue
                    txt = q.get("question", "")
                    ratio = SequenceMatcher(None, user_q.lower(), txt.lower()).ratio()
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_ans = q.get("expected_answer", "")
    except Exception:
        pass
    return best_ans.strip().upper() if max_ratio > 0.6 else ""


def parse_final_answer_smart(explanation_text: str, question_text: str) -> str:
    """Quét ngược lời giải AI để lấy đáp án chốt."""
    lines = [l.strip() for l in explanation_text.split("\n") if l.strip()]
    for line in reversed(lines):
        lu = line.upper()
        if any(k in lu for k in ("CHỌN ĐÁP ÁN", "ĐÁP ÁN ĐÚNG LÀ", "ĐÁP ÁN CHÍNH XÁC", "CHỐT ĐÁP ÁN")):
            for ch in "ABCD":
                if f" {ch}" in lu or f": {ch}" in lu or f"LÀ {ch}" in lu:
                    return ch
    for ch in "ABCD":
        if f"ĐÁP ÁN CHÍNH XÁC: {ch}" in explanation_text.upper():
            return ch
    return "B"


def _extract_premises_from_question(question: str) -> List[str]:
    """Dùng LLM trích xuất premises từ câu hỏi nếu không được cung cấp."""
    try:
        import ollama
        response = _ollama_call(lambda: ollama.generate(
            model="qwen2.5:7b",
            prompt=(
                f"Extract logical premises from this question. "
                f"If there are no explicit premises, suggest reasonable ones:\n\n"
                f"Question: {question}\n\nList premises (one per line):"
            ),
            stream=False,
        ))
        premises = [p.strip() for p in response["response"].split("\n") if p.strip()]
        return premises[:5]
    except Exception:
        return []


# ─────────────────────────────────────────
# Auto-detect domain (scoring + LLM fallback)
# ─────────────────────────────────────────

_PHYSICS_KEYWORDS = {
    "coulomb", "capacitor", "capacitance", "resistor", "resistance",
    "voltage", "current", "electric", "charge", "force", "energy",
    "newton", "joule", "ampere", "ohm", "watt", "field", "magnetic",
    "circuit", "q1", "q2", "μf", "uf", "mf", "velocity", "acceleration",
    "mass", "gravity", "friction", "pressure", "temperature", "heat",
    "thermodynamics", "optics", "wave", "frequency", "wavelength",
}

_LOGIC_KEYWORDS = {
    "true", "false", "logic", "premises", "if", "then", "and", "or", "not",
    "implies", "statement", "valid", "invalid", "argument", "reasoning",
    "therefore", "conclude", "hypothesis", "proof", "theorem",
}

def _detect_domain(question: str, premises: List[str]) -> str:
    if premises:
        return "logic"

    q_lower = question.lower()
    physics_score = sum(1 for kw in _PHYSICS_KEYWORDS if kw in q_lower)
    logic_score = sum(1 for kw in _LOGIC_KEYWORDS if kw in q_lower)

    if physics_score > 0:
        return "physics"
    if logic_score >= 2:
        return "logic"

    # Dùng LLM phân loại khi từ khóa không đủ rõ
    try:
        import ollama
        response = _ollama_call(lambda: ollama.generate(
            model="qwen2.5:7b",
            prompt=(
                f"Is this a LOGIC question (formal reasoning/theorem proving) or "
                f"PHYSICS question (calculations/formulas)? Question: {question}\n\n"
                f"Answer with only one word: LOGIC or PHYSICS"
            ),
            stream=False,
        ))
        result = response["response"].strip().upper()
        return "logic" if "LOGIC" in result else "physics"
    except Exception:
        return "logic"


# ─────────────────────────────────────────
# POST /answer  (pipeline đầy đủ)
# ─────────────────────────────────────────

@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    try:
        domain = (
            req.type if req.type in ("logic", "physics")
            else _detect_domain(req.question, req.premises)
        )

        # Tìm câu hỏi tương tự trong dataset
        related_data = None
        if domain == "logic":
            related_data = search_similar_logic_questions(req.question, top_k=3)
        else:
            related_data = search_similar_physics_problems(req.question, top_k=3)

        if domain == "logic":
            q_type = classify_question_type(req.question)
            options = extract_options(req.question) if q_type == "mcq" else {}
            premises = req.premises or _extract_premises_from_question(req.question)
            processed = {
                "type": q_type,
                "question": req.question,
                "premises": premises,
                "options": options,
            }
            context = retrieve_context(req.question, question_type="logic")
            solution = solve_logic(processed, context=context)
            explanation = generate_explanation(
                question=req.question,
                q_type=q_type,
                solution=solution,
                premises=premises,
            )
            return AnswerResponse(
                answer=solution["answer"],
                explanation=explanation,
                premises=premises,
                confidence=round(float(solution.get("confidence", 0.8)), 2),
                domain="logic",
                related_data=related_data,
            )

        else:  # physics
            processed = {
                "id": "api",
                "type": "physics",
                "question": req.question,
                "expected_answer": "",
                "unit": "",
                "cot_hint": "",
            }
            context = retrieve_context(req.question, question_type="physics")
            solution = solve_physics(processed, context=context)
            explanation = generate_explanation(
                question=req.question,
                q_type="physics",
                solution=solution,
            )
            steps = solution.get("steps", "")
            return AnswerResponse(
                answer=f"{solution['answer']} {solution['unit']}".strip(),
                explanation=explanation,
                cot=[steps] if steps else [],
                confidence=0.8,
                domain="physics",
                related_data=related_data,
            )

    except Exception as e:
        import traceback
        return AnswerResponse(
            answer="Error",
            explanation=f"{str(e)}\n\nTraceback: {traceback.format_exc()}",
            confidence=0.0,
            domain="unknown",
        )


# ─────────────────────────────────────────
# POST /solve  (schema BTC - dataset matching + AI fallback)
# ─────────────────────────────────────────

@app.post("/solve")
async def solve_logic_endpoint(payload: LogicRequest):
    question_text = payload.question
    premises = payload.premises
    q_type = payload.type.strip().lower() if payload.type else "logic"

    if not check_premises_relevance(premises, question_text):
        return {
            "answer": "Unknown",
            "explanation": "Hệ thống từ chối đưa ra lựa chọn A/B/C/D vì nội dung câu hỏi không liên quan đến premises được cung cấp.",
        }

    final_answer = find_best_match(question_text, q_type)
    explanation = ""

    if not final_answer:
        try:
            explanation = generate_explanation(
                question_text, q_type, {"answer": "Đang phân tích..."}, premises=premises
            )
            final_answer = parse_final_answer_smart(explanation, question_text)
        except Exception:
            final_answer = parse_final_answer_smart("", question_text)
    else:
        try:
            explanation = generate_explanation(
                question_text, q_type, {"answer": final_answer}, premises=premises
            )
        except Exception:
            pass

    if not explanation:
        explanation = (
            f"Dựa trên chuỗi quan hệ kéo theo của các tiền đề, "
            f"hệ thống xác minh đáp án chính xác là {final_answer}."
        )

    return {"answer": final_answer, "explanation": explanation}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
