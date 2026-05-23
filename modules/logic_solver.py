"""
logic_solver.py — Giải câu hỏi logic bằng Z3 + LLM
Người phụ trách: Member 3
"""

import re
import concurrent.futures
import ollama
from z3 import (Bool, BoolSort, BoolVal, Const, Function, IntSort,
                Solver, Not, And, Or, Implies, ForAll, Exists, sat, unknown)

FOL_KEYWORDS = frozenset({"ForAll", "Exists", "Implies", "And", "Or", "→", "->"})

_LLM_TIMEOUT = 60  # seconds — Ollama calls abort after this


def _ollama_call(fn):
    """Chạy fn() với timeout; ném TimeoutError nếu quá _LLM_TIMEOUT giây."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn)
        return future.result(timeout=_LLM_TIMEOUT)


def check_statement_with_llm(premises: list, statement: str, context: str = "") -> str:
    """
    Dùng LLM để kiểm tra logic khi Z3 không parse được FOL phức tạp
    Trả về: 'Yes', 'No', hoặc 'Unknown'
    """
    premises_text = "\n".join(f"- {p}" for p in premises)

    context_block = f"Supplementary context:\n{context}\n\n" if context else ""
    prompt = f"""{context_block}You are a strict logic evaluator.

Given these premises:
{premises_text}

Does this statement logically follow?
Statement: {statement}

Rules:
- Answer ONLY based on the given premises
- If premises are insufficient to conclude → answer Unknown
- Do NOT use outside knowledge

Reply with EXACTLY one word: Yes, No, or Unknown"""

    response = _ollama_call(lambda: ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    ))

    result = response["message"]["content"].strip().lower()
    if "unknown" in result:
        return "Unknown"
    if result.startswith("yes"):
        return "Yes"
    if result.startswith("no"):
        return "No"
    return "Unknown"


def _looks_like_fol(premises: list) -> bool:
    combined = " ".join(premises)
    return any(kw in combined for kw in FOL_KEYWORDS)


def _normalize_fol(text: str) -> str:
    # ForAll(x, ...) → ForAll([x], ...)  — Z3 requires a list of bound vars
    text = re.sub(r'\bForAll\((\w+),', r'ForAll([\1],', text)
    text = re.sub(r'\bExists\((\w+),', r'Exists([\1],', text)
    # P(x) → Q(x)  →  Implies(P(x), Q(x))
    arrow = r'([A-Za-z_]\w*(?:\([^()]*\))?)\s*(?:→|->)\s*([A-Za-z_]\w*(?:\([^()]*\))?)'
    text = re.sub(arrow, r'Implies(\1, \2)', text)
    return text


def _build_z3_namespace(all_texts: list) -> dict:
    combined = " ".join(all_texts)
    predicates = set(re.findall(r'\b([a-z_]\w*)\s*\(', combined))
    z3_keywords = {"ForAll", "Exists", "Implies", "And", "Or", "Not", "True", "False"}
    predicates -= {k.lower() for k in z3_keywords}

    # Capture all non-predicate identifiers: lowercase bound vars (x, y) AND
    # capitalized ground constants (Sophia, John) — both become Const in Z3
    all_terms = set(re.findall(r'\b([A-Za-z_]\w*)\b(?!\s*\()', combined))
    all_terms -= z3_keywords
    all_terms -= predicates

    ns = {
        "ForAll": ForAll, "Exists": Exists, "Implies": Implies,
        "And": And, "Or": Or, "Not": Not,
        "True": BoolVal(True), "False": BoolVal(False),
        "__builtins__": {},
    }
    for t in all_terms:
        ns[t] = Const(t, IntSort())
    for p in predicates:
        ns[p] = Function(p, IntSort(), BoolSort())
    return ns


def _parse_fol_to_z3(text: str, ns: dict):
    try:
        return eval(_normalize_fol(text), {"__builtins__": {}}, ns)
    except Exception:
        return None


def _solve_fol(premises: list, question: str):
    ns = _build_z3_namespace(premises + [question])
    z3_premises = [_parse_fol_to_z3(p, ns) for p in premises]
    if any(e is None for e in z3_premises):
        return None  # unparseable premise → fall through to LLM
    q_expr = _parse_fol_to_z3(question, ns)
    if q_expr is None:
        return None

    solver = Solver()
    for expr in z3_premises:
        solver.add(expr)

    solver.push()
    solver.add(Not(q_expr))
    if solver.check() != sat:
        solver.pop()
        return "Yes"
    solver.pop()

    solver.push()
    solver.add(q_expr)
    if solver.check() != sat:
        solver.pop()
        return "No"
    solver.pop()
    return None


def _solve_propositional(premises: list, question: str):
    solver = Solver()
    all_stmts = premises + [question]
    var_map = {s: Bool(f"p{i}") for i, s in enumerate(all_stmts)}
    for p in premises:
        solver.add(var_map[p])
    q_var = var_map[question]

    solver.push()
    solver.add(Not(q_var))
    if solver.check() != sat:
        solver.pop()
        return "Yes"
    solver.pop()

    solver.push()
    solver.add(q_var)
    if solver.check() != sat:
        solver.pop()
        return "No"
    solver.pop()
    return None


def solve_with_z3(premises: list, question: str):
    """
    Routes to FOL solver if premises contain ForAll/Exists/Implies/→,
    otherwise falls back to propositional Bool-variable mapping.
    Returns 'Yes'/'No' if Z3 can decide, else None.
    """
    try:
        if _looks_like_fol(premises + [question]):
            return _solve_fol(premises, question)
        return _solve_propositional(premises, question)
    except Exception:
        return None


def solve_yesno(premises: list, question: str, context: str = "") -> dict:
    """
    Giải câu hỏi Yes/No
    Input:  premises (list), question (str)
    Output: {answer, confidence, method}
    """
    # Step 1: Try Z3 first
    z3_answer = solve_with_z3(premises, question)
    if z3_answer is not None:
        return {"answer": z3_answer, "confidence": 0.95, "method": "z3"}

    # Step 2: LLM fallback
    answer = check_statement_with_llm(premises, question, context)
    return {
        "answer": answer,
        "confidence": 0.70 if answer != "Unknown" else 0.50,
        "method": "llm_logic"
    }


def solve_mcq(premises: list, question: str, options: dict, context: str = "") -> dict:
    """
    Giải câu hỏi trắc nghiệm A/B/C/D
    Input:  premises (list), question (str), options (dict)
    Output: {answer, confidence, reasoning}
    """
    if not options:
        return {"answer": "Unknown", "confidence": 0.50, "reasoning": "No options found"}

    premises_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises))
    options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

    context_block = f"Supplementary context:\n{context}\n\n" if context else ""

    response = _ollama_call(lambda: ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict logic evaluator. "
                    "Your ONLY job is to check if the premises logically PROVE an answer choice. "
                    "Never guess. Never use outside knowledge. "
                    "When in doubt, output Uncertain."
                )
            },
            {
                "role": "user",
                "content": (
                    f"{context_block}"
                    f"PREMISES:\n{premises_text}\n\n"
                    f"QUESTION: {question}\n\n"
                    f"CHOICES:\n{options_text}\n\n"
                    "RULES:\n"
                    "- Carefully check whether the premises CLEARLY and DIRECTLY PROVE each choice\n"
                    "- If and ONLY IF the premises CONCLUSIVELY prove exactly ONE choice → return that letter (A/B/C/D)\n"
                    "- If the premises do NOT clearly and directly support any single answer, you MUST output: Uncertain\n"
                    "- Do not guess. Do not pick the most likely answer. Do not use outside knowledge.\n"
                    "- Only output a letter if the premises logically PROVE that answer beyond any doubt\n\n"
                    "ANSWER (one word only — A, B, C, D, or Uncertain):"
                )
            }
        ],
        options={"temperature": 0.0},
    ))

    content = response["message"]["content"].strip()

    # Uncertain/Unknown/anything unexpected → "Unknown" to match dataset expected values
    tokens = content.split()
    first = tokens[0].upper() if tokens else "UNCERTAIN"
    valid_answers = {"A", "B", "C", "D"}
    answer = first if first in valid_answers else "Unknown"

    return {
        "answer": answer,
        "confidence": 0.70 if answer != "Unknown" else 0.50,
        "reasoning": content
    }


def solve_logic(processed_question: dict, context: str = "") -> dict:
    """
    Hàm chính — nhận câu hỏi đã xử lý, trả về kết quả
    Input:  dict từ classifier.process_logic_entry()
    Output: {answer, confidence, method, reasoning}
    """
    q_type = processed_question["type"]
    premises = processed_question["premises"]
    question = processed_question["question"]
    options = processed_question.get("options", {})

    if q_type == "yesno":
        result = solve_yesno(premises, question, context)
    else:
        result = solve_mcq(premises, question, options, context)

    return result


# ── Test thử ──
if __name__ == "__main__":
    print("Test logic_solver.py...")

    test_yesno = {
        "type": "yesno",
        "premises": [
            "If a student completes all required courses, they are eligible for graduation.",
            "If eligible for graduation and GPA > 3.0, they receive honors.",
            "John completed all required courses.",
            "John has GPA 3.5."
        ],
        "question": "Is John invited to the graduation ceremony?",
        "options": {}
    }

    test_mcq = {
        "type": "mcq",
        "premises": [
            "All students who pass the entrance exam are admitted.",
            "All admitted students complete orientation.",
            "Students who complete orientation get a student ID.",
            "Students with ID can access the library.",
            "Maria passed the entrance exam."
        ],
        "question": "Which is the strongest conclusion about Maria?",
        "options": {
            "A": "Maria can access the library",
            "B": "Maria needs to apply separately for ID",
            "C": "Maria is not admitted",
            "D": "Maria skipped orientation"
        }
    }

    print("\n── Test Yes/No ──")
    r1 = solve_logic(test_yesno)
    print(f"Answer    : {r1['answer']}")
    print(f"Confidence: {r1['confidence'] * 100:.0f}%")

    print("\n── Test MCQ ──")
    r2 = solve_logic(test_mcq)
    print(f"Answer    : {r2['answer']}")
    print(f"Confidence: {r2['confidence'] * 100:.0f}%")
    print(f"Reasoning : {r2['reasoning']}")

    print("\n✅ logic_solver.py hoạt động!")
