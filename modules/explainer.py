"""
explainer.py — Sinh giải thích tự nhiên chất lượng cao
Người phụ trách: Member 5
"""

import concurrent.futures
import ollama

_LLM_TIMEOUT = 60

def _ollama_call(fn):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=_LLM_TIMEOUT)


# ============================================================
# FEW-SHOT EXAMPLES — Học từ dataset thật
# ============================================================

LOGIC_EXAMPLES = """
EXAMPLE 1 (Yes/No):
Premises:
1. If a student completes all courses, they are eligible to graduate.
2. If eligible and GPA > 3.0, they receive honors.
3. John completed all courses. John has GPA 3.5.
Question: Does John receive honors?
Solution: answer=Yes

ANSWER: Yes
REASONING:
- Premise 3: John completed all courses → satisfies premise 1
- Premise 1: completed courses → eligible to graduate ✅
- Premise 3: John GPA 3.5 > 3.0 → satisfies premise 2 condition
- Premise 2: eligible ∧ GPA>3.0 → receives honors ✅
CONCLUSION: John receives honors because he satisfies both conditions in premise 2.

---

EXAMPLE 2 (MCQ - Unknown):
Premises:
1. All Python code is well-tested.
2. If well-tested, the project is optimized.
Question: Which conclusion needs fewest premises?
A. If not optimized, then not well-tested
B. If optimized, then well-structured
Options checked: A needs premises 1,2 via contrapositive. B cannot be derived.
Solution: answer=Unknown

ANSWER: Unknown
REASONING:
- Option A: Contrapositive of premise 2 — valid but needs 2 premises
- Option B: Cannot be derived from given premises ❌
- Option C: Requires additional premises not given ❌
- Option D: Contradicts premise 1 ❌
CONCLUSION: No single option can be confirmed with certainty from the given premises alone.
"""

PHYSICS_EXAMPLES = """
EXAMPLE 1 (Capacitor Energy):
Question: Calculate energy stored when C = 100 μF, U = 30 V.
Solution: formula=W=0.5CU², answer=45, unit=J

ANSWER: W = 45 J
REASONING:
- Formula: W = ½ × C × U²
- Given: C = 100 μF = 100×10⁻⁶ F, U = 30 V
- Step 1: W = 0.5 × 100×10⁻⁶ × 30²
- Step 2: W = 0.5 × 100×10⁻⁶ × 900
- Step 3: W = 45×10⁻³ J = ... [check units]
CONCLUSION: The capacitor stores 45 J of energy.

---

EXAMPLE 2 (Coulomb Force):
Question: F between q1=6×10⁻⁸ C, q2=-6×10⁻⁸ C, r=8 cm.
Solution: formula=F=k|q1q2|/r², answer=0.05, unit=N

ANSWER: F = 0.05 N
REASONING:
- Formula: F = k × |q1 × q2| / r²
- Given: k=9×10⁹, q1=6×10⁻⁸ C, q2=6×10⁻⁸ C, r=0.08 m
- Step 1: F = 9×10⁹ × (6×10⁻⁸ × 6×10⁻⁸) / (0.08)²
- Step 2: F = 9×10⁹ × 36×10⁻¹⁶ / 6.4×10⁻³
- Step 3: F = 0.05 N
CONCLUSION: The Coulomb force between the two charges is 0.05 N.
"""


# ============================================================
# HÀM SINH GIẢI THÍCH
# ============================================================

def explain_logic(question: str, premises: list,
                  solution: dict, q_type: str) -> str:
    """
    Sinh giải thích cho câu hỏi logic
    Input:  question, premises, solution từ logic_solver
    Output: string có format ANSWER + REASONING + CONCLUSION
    """
    premises_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises))
    answer = solution.get("answer", "Unknown")
    reasoning = solution.get("reasoning", "")

    prompt = f"""
{LOGIC_EXAMPLES}

---
NOW EXPLAIN THIS:
Premises:
{premises_text}

Question: {question}
Calculated answer: {answer}
{f'Reasoning hint: {reasoning}' if reasoning else ''}

Write explanation in EXACTLY the same format as examples above.
- Use specific premise numbers
- Be concise (40-70 words for REASONING)
- End with clear CONCLUSION sentence
"""

    response = _ollama_call(lambda: ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": """You are an expert logic tutor writing clear explanations.
RULES:
- Reference premises by number (e.g. 'Premise 1')
- Use exact numbers, never approximate
- Follow format strictly: ANSWER / REASONING / CONCLUSION
- Never add text outside the format"""
            },
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.2},
    ))

    return response["message"]["content"].strip()


def explain_physics(question: str, solution: dict) -> str:
    """
    Sinh giải thích cho bài toán vật lý
    Input:  question, solution từ physics_solver
    Output: string có format ANSWER + REASONING + CONCLUSION
    """
    answer = solution.get("answer", "")
    unit = solution.get("unit", "")
    formula = solution.get("formula", "")
    given = solution.get("given", "")
    steps = solution.get("steps", "")

    prompt = f"""
{PHYSICS_EXAMPLES}

---
NOW EXPLAIN THIS:
Question: {question}
Formula used: {formula}
Given values: {given}
Calculation steps: {steps}
Final answer: {answer} {unit}

Write explanation in EXACTLY the same format as examples above.
- Show formula clearly
- Show each calculation step with numbers
- End with CONCLUSION stating the final answer
"""

    response = _ollama_call(lambda: ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": """You are a physics tutor writing precise explanations.
RULES:
- Always show the formula first
- Include all given values with units
- Show each calculation step
- Use scientific notation where appropriate
- Follow format: ANSWER / REASONING / CONCLUSION"""
            },
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.1},
    ))

    return response["message"]["content"].strip()


def generate_explanation(question: str, q_type: str,
                         solution: dict, premises: list = None) -> str:
    """
    Hàm chính — tự động chọn đúng hàm giải thích
    Input:  question, type ('logic'/'physics'), solution, premises
    Output: explanation string
    """
    if q_type in ["yesno", "mcq", "logic"]:
        return explain_logic(question, premises or [], solution, q_type)
    else:
        return explain_physics(question, solution)


# ── Test thử ──
if __name__ == "__main__":
    print("Test explainer.py...")

    # Test logic
    print("\n── Test Logic Explanation ──")
    logic_result = explain_logic(
        question="Is John eligible to graduate?",
        premises=[
            "If a student completes all courses, they are eligible to graduate.",
            "John has completed all required courses.",
            "John has a GPA of 3.5."
        ],
        solution={"answer": "Yes", "confidence": 90},
        q_type="yesno"
    )
    print(logic_result)

    # Test physics
    print("\n── Test Physics Explanation ──")
    physics_result = explain_physics(
        question="Calculate the energy stored in capacitor C = 100 μF, U = 30 V.",
        solution={
            "answer": "45",
            "unit": "J",
            "formula": "W = 0.5 × C × U²",
            "given": "C = 100 μF, U = 30 V",
            "steps": "W = 0.5 × 100×10⁻⁶ × 900 = 0.045 J"
        }
    )
    print(physics_result)

    print("\n✅ explainer.py hoạt động!")
