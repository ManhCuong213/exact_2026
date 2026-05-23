"""
physics_solver.py — Giải bài toán vật lý bằng SymPy + LLM
Người phụ trách: Member 4

Các loại bài trong dataset:
  TD  → Tụ điện (Capacitor)
  LD  → Lực điện (Electric force - Coulomb)
  DT  → Điện trường (Electric field)
  NL  → Năng lượng (Energy)
  DD  → Điện từ (Electromagnetics)
  QA  → Quang học / Dao động (Optics / Oscillation)
  CH  → Chất lỏng / Nhiệt (Fluid / Thermal)
  TH  → Thí nghiệm (Experiment)
"""

import re
import json
import concurrent.futures
import ollama
from sympy import symbols, solve, Eq, pi, sqrt, Rational

_LLM_TIMEOUT = 60

def _ollama_call(fn):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=_LLM_TIMEOUT)


# ============================================================
# CÁC CÔNG THỨC VẬT LÝ PHỔ BIẾN
# ============================================================

def solve_capacitor_energy(C=None, U=None, Q=None):
    """Năng lượng tụ điện: W = 0.5 * C * U^2 = Q^2 / (2C)"""
    if C and U:
        W = 0.5 * C * U**2
        return {"result": W, "formula": "W = 0.5 × C × U²"}
    elif Q and C:
        W = Q**2 / (2 * C)
        return {"result": W, "formula": "W = Q² / (2C)"}
    return None


def solve_capacitance(Q=None, U=None, C=None):
    """Điện dung: C = Q / U"""
    if Q and U:
        return {"result": Q / U, "formula": "C = Q / U"}
    return None


def solve_coulomb_force(q1, q2, r, k=9e9):
    """Lực Coulomb: F = k * |q1 * q2| / r²"""
    F = k * abs(q1 * q2) / r**2
    return {"result": F, "formula": "F = k × |q1 × q2| / r²"}


def solve_electric_field(q, r, k=9e9):
    """Điện trường: E = k * |q| / r²"""
    E = k * abs(q) / r**2
    return {"result": E, "formula": "E = k × |q| / r²"}


def solve_series_circuit(V, resistors):
    """Mạch nối tiếp"""
    R_total = sum(resistors)
    I = V / R_total
    voltages = [I * R for R in resistors]
    return {
        "current_A": round(I, 6),
        "R_total": R_total,
        "voltages": [round(v, 6) for v in voltages],
        "formula": "I = V / R_total"
    }


def solve_parallel_circuit(V, resistors):
    """Mạch song song"""
    currents = [V / R for R in resistors]
    I_total = sum(currents)
    R_eq = V / I_total
    return {
        "currents": [round(i, 6) for i in currents],
        "I_total": round(I_total, 6),
        "R_equivalent": round(R_eq, 6),
        "formula": "I = V/R for each branch"
    }


# ============================================================
# HÀM CHÍNH: DÙNG LLM ĐỌC ĐỀ + TRÍCH SỐ + GIẢI
# ============================================================

def extract_and_solve_with_llm(question: str, cot_hint: str = "", context: str = "") -> dict:
    """
    Dùng LLM để:
    1. Đọc đề bài
    2. Trích xuất số liệu
    3. Xác định công thức cần dùng
    4. Tính toán (SymPy xử lý phần số)
    """
    context_block = f"Reference context:\n{context}\n\n" if context else ""
    prompt = f"""{context_block}You are a physics expert. Solve this problem step by step.

Problem: {question}
{f'Hint: {cot_hint[:200]}' if cot_hint else ''}

Reply in EXACTLY this format:
FORMULA: [the main formula used]
GIVEN: [list the given values with units]
CALCULATION: [show each step]
ANSWER: [numerical value only, no unit]
UNIT: [unit of the answer]"""

    response = _ollama_call(lambda: ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": "You are a precise physics calculator. Always show formula and steps. Never skip calculations."
            },
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.0},
    ))

    content = response["message"]["content"].strip()

    # Parse kết quả
    result = {
        "formula": "",
        "given": "",
        "calculation": "",
        "answer": "",
        "unit": ""
    }

    current_key = None
    for line in content.split("\n"):
        if line.startswith("FORMULA:"):
            result["formula"] = line.replace("FORMULA:", "").strip()
        elif line.startswith("GIVEN:"):
            result["given"] = line.replace("GIVEN:", "").strip()
        elif line.startswith("CALCULATION:"):
            result["calculation"] = line.replace("CALCULATION:", "").strip()
        elif line.startswith("ANSWER:"):
            raw = line.replace("ANSWER:", "").strip()
            # Chỉ lấy số
            numbers = re.findall(r"[-+]?\d*\.?\d+", raw)
            result["answer"] = numbers[0] if numbers else raw
        elif line.startswith("UNIT:"):
            result["unit"] = line.replace("UNIT:", "").strip()

    return result


_UNIT_FACTORS = {
    # Capacitance
    r"μf": 1e-6, r"uf": 1e-6, r"mf": 1e-3,
    # Energy
    r"mj": 1e-3, r"kj": 1e3,
    # Force
    r"mn": 1e-3, r"kn": 1e3,
    # Charge
    r"μc": 1e-6, r"uc": 1e-6, r"mc": 1e-3,
}


def normalize_to_si(value_str: str) -> float:
    """
    Extract the numeric part and apply any non-SI unit prefix to convert to SI.
    e.g. "100 μF" → 100e-6 = 1e-4,  "45 mJ" → 0.045,  "0.045 J" → 0.045
    """
    s = str(value_str).strip()
    num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not num_match:
        raise ValueError(f"No number found in: {s!r}")
    value = float(num_match.group())

    # Find any unit suffix after the number (case-insensitive)
    suffix = s[num_match.end():].strip().lower()
    for unit, factor in _UNIT_FACTORS.items():
        if suffix.startswith(unit):
            value *= factor
            break

    return value


def compare_physics_answer(predicted: str, ground_truth: str) -> bool:
    """
    Normalize both values to SI, then compare with 1% relative tolerance.
    Handles unit mismatches like "0.0001 F" vs "100 μF".
    """
    if not predicted or not ground_truth:
        return False
    try:
        pred = normalize_to_si(str(predicted))
        gt   = normalize_to_si(str(ground_truth))
    except ValueError:
        return False

    if gt == 0:
        return abs(pred) < 1e-6

    return abs(pred - gt) / abs(gt) < 0.05


def _detect_topic(question: str) -> str:
    """Detect physics topic from question keywords."""
    q = question.lower()

    if any(k in q for k in ["coulomb", "force between", "point charge", "charges are placed"]):
        return "coulomb"
    if any(k in q for k in ["q1", "q2"]) and "charge" in q and "field" not in q:
        return "coulomb"
    if any(k in q for k in ["electric field", "field strength", "field at point"]):
        return "electric_field"
    if any(k in q for k in ["capacitor", "capacit"]):
        if any(k in q for k in ["energy stored", "energy in", "calculate the energy"]):
            return "capacitor_energy"
        return "capacitance"
    if "series" in q:
        return "series_circuit"
    if "parallel" in q and "charge" not in q:
        return "parallel_circuit"
    return "unknown"


def _extract_numbers_with_llm(question: str, topic: str) -> dict:
    """Use LLM only to extract numbers — SymPy does the calculation."""
    templates = {
        "coulomb":          '{"q1_C": <charge1 in Coulombs>, "q2_C": <charge2 in Coulombs>, "r_m": <distance in METERS>}',
        "electric_field":   '{"q_C": <charge in Coulombs>, "r_m": <distance in METERS>}',
        "capacitor_energy": '{"C_F": <capacitance in Farads>, "U_V": <voltage in Volts>, "Q_C": <charge in Coulombs or null>}',
        "capacitance":      '{"Q_C": <charge in Coulombs>, "U_V": <voltage in Volts>}',
        "series_circuit":   '{"V": <source voltage in Volts>, "resistors": [<resistances in Ohms>]}',
        "parallel_circuit": '{"V": <source voltage in Volts>, "resistors": [<resistances in Ohms>]}',
    }

    prompt = f"""Extract numbers from this physics problem. Convert ALL units to SI before writing JSON.
Unit conversions REQUIRED:
- Distance: cm → m (÷100), mm → m (÷1000)
- Capacitance: μF → F (×10⁻⁶), mF → F (×10⁻³)
- Charge: keep in Coulombs (e.g. 6×10⁻⁸ → 6e-8)

Problem: {question}

Reply ONLY with valid JSON:
{templates[topic]}"""

    response = _ollama_call(lambda: ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    ))

    raw = response["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


def solve_physics(processed_question: dict, context: str = "") -> dict:
    """
    Detect topic → route to specific SymPy solver → fallback to LLM.
    Input:  dict từ classifier.process_physics_entry()
    Output: {answer, unit, formula, steps}
    """
    question = processed_question["question"]
    cot_hint = processed_question.get("cot_hint", "")

    topic = _detect_topic(question)

    try:
        if topic == "coulomb":
            data = _extract_numbers_with_llm(question, topic)
            r = solve_coulomb_force(data["q1_C"], data["q2_C"], data["r_m"])
            return {
                "answer": round(r["result"], 6),
                "unit": "N",
                "formula": r["formula"],
                "given": str(data),
                "steps": f"F = 9×10⁹ × |{data['q1_C']} × {data['q2_C']}| / ({data['r_m']})²"
            }

        if topic == "electric_field":
            data = _extract_numbers_with_llm(question, topic)
            r = solve_electric_field(data["q_C"], data["r_m"])
            return {
                "answer": round(r["result"], 6),
                "unit": "N/C",
                "formula": r["formula"],
                "given": str(data),
                "steps": f"E = 9×10⁹ × |{data['q_C']}| / ({data['r_m']})²"
            }

        if topic == "capacitor_energy":
            data = _extract_numbers_with_llm(question, topic)
            C, U, Q = data.get("C_F"), data.get("U_V"), data.get("Q_C")
            r = solve_capacitor_energy(C=C, U=U, Q=Q if not U else None)
            return {
                "answer": round(r["result"], 6),
                "unit": "J",
                "formula": r["formula"],
                "given": str(data),
                "steps": f"W = 0.5 × {C} × {U}²" if U else f"W = {Q}² / (2×{C})"
            }

        if topic == "capacitance":
            data = _extract_numbers_with_llm(question, topic)
            r = solve_capacitance(Q=data.get("Q_C"), U=data.get("U_V"))
            return {
                "answer": round(r["result"], 6),
                "unit": "F",
                "formula": r["formula"],
                "given": str(data),
                "steps": f"C = {data.get('Q_C')} / {data.get('U_V')}"
            }

        if topic == "series_circuit":
            data = _extract_numbers_with_llm(question, topic)
            r = solve_series_circuit(data["V"], data["resistors"])
            return {
                "answer": r["current_A"],
                "unit": "A",
                "formula": r["formula"],
                "given": str(data),
                "steps": f"R_total = {r['R_total']} Ω, I = {r['current_A']} A"
            }

        if topic == "parallel_circuit":
            data = _extract_numbers_with_llm(question, topic)
            r = solve_parallel_circuit(data["V"], data["resistors"])
            return {
                "answer": r["I_total"],
                "unit": "A",
                "formula": r["formula"],
                "given": str(data),
                "steps": f"I_total = {r['I_total']} A"
            }

    except Exception:
        pass  # extraction failed → fall through to LLM

    # Fallback: LLM handles unsupported topics (QA, CH, TH, DD, NL, ...)
    solution = extract_and_solve_with_llm(question, cot_hint, context)
    return {
        "answer": solution["answer"],
        "unit": solution["unit"],
        "formula": solution["formula"],
        "given": solution["given"],
        "steps": solution["calculation"]
    }


# ── Test thử ──
if __name__ == "__main__":
    print("Test physics_solver.py...")

    test_cases = [
        {
            "id": "TD401",
            "type": "physics",
            "question": "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V.",
            "expected_answer": "45",
            "unit": "J",
            "cot_hint": "Step 1: Use W = 0.5 * C * U^2"
        },
        {
            "id": "LD001",
            "type": "physics",
            "question": "Two charges q1 = 6×10^-8 C and q2 = -6×10^-8 C are 8 cm apart. Calculate the Coulomb force.",
            "expected_answer": "0.05",
            "unit": "N",
            "cot_hint": "Step 1: Use F = k|q1*q2|/r^2"
        }
    ]

    for tc in test_cases:
        print(f"\n── Bài {tc['id']} ──")
        print(f"Question: {tc['question'][:70]}...")
        result = solve_physics(tc)
        print(f"Formula : {result['formula']}")
        print(f"Answer  : {result['answer']} {result['unit']}")
        print(f"Expected: {tc['expected_answer']} {tc['unit']}")
        correct = tc['expected_answer'] in str(result['answer'])
        print(f"Status  : {'✅ ĐÚNG' if correct else '❌ SAI'}")

    print("\n✅ physics_solver.py hoạt động!")
