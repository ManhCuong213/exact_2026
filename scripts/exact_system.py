import ollama
import json
from sympy import symbols, Eq, solve
from z3 import Real, Solver, sat

# ============================================================
# MODULE 1: PHÂN LOẠI CÂU HỎI
# ============================================================
def classify_question(question: str) -> str:
    """
    Phân loại câu hỏi thành: 'logic' hoặc 'circuit'
    Trả về đúng 1 trong 2 từ đó
    """
    prompt = f"""Classify this question into exactly one category.

Question: {question}

Categories:
- "logic": questions about rules, eligibility, conditions, policies, 
           whether someone qualifies for something
- "circuit": questions about electrical circuits, resistors, voltage, 
             current, Ohm's law, series/parallel circuits

Reply with ONLY one word: logic OR circuit
"""
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response["message"]["content"].strip().lower()
    
    # Đảm bảo chỉ trả về 2 giá trị hợp lệ
    if "circuit" in result:
        return "circuit"
    return "logic"


# ============================================================
# MODULE 2: TRÍCH XUẤT DỮ LIỆU TỪ CÂU HỎI
# ============================================================
def extract_data(question: str, q_type: str) -> dict:
    """
    Dùng LLM trích xuất số liệu từ câu hỏi
    Trả về dict chứa các giá trị cần thiết
    """
    
    if q_type == "logic":
        prompt = f"""Extract data from this logic question. Reply ONLY with valid JSON.

Question: {question}

Extract:
{{
  "gpa": <number or null>,
  "credits": <number or null>,
  "min_gpa": <required minimum GPA or null>,
  "min_credits": <required minimum credits or null>
}}"""

    else:  # circuit
        prompt = f"""Extract data from this circuit question. Reply ONLY with valid JSON.

Question: {question}

Extract:
{{
  "voltage": <source voltage as number>,
  "resistors": [<list of resistance values in ohms>],
  "circuit_type": "series" or "parallel"
}}"""

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw = response["message"]["content"].strip()
    
    # Làm sạch JSON (xóa ```json nếu có)
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(raw)
    except:
        return {}


# ============================================================
# MODULE 3A: GIẢI LOGIC BẰNG Z3
# ============================================================
def solve_logic(data: dict) -> dict:
    """Dùng Z3 giải bài toán logic"""
    
    try:
        gpa = Real('gpa')
        credits = Real('credits')
        
        solver = Solver()
        
        # Thêm quy tắc
        if data.get("min_gpa"):
            solver.add(gpa >= data["min_gpa"])
        if data.get("min_credits"):
            solver.add(credits >= data["min_credits"])
        
        # Thêm dữ liệu thực tế
        if data.get("gpa"):
            solver.add(gpa == data["gpa"])
        if data.get("credits"):
            solver.add(credits == data["credits"])
        
        result = solver.check()
        eligible = (result == sat)
        
        # Phân tích từng điều kiện
        reasons = []
        if data.get("gpa") and data.get("min_gpa"):
            if data["gpa"] >= data["min_gpa"]:
                reasons.append(f"GPA {data['gpa']} >= {data['min_gpa']} ✅")
            else:
                reasons.append(f"GPA {data['gpa']} < {data['min_gpa']} ❌")
                
        if data.get("credits") and data.get("min_credits"):
            if data["credits"] >= data["min_credits"]:
                reasons.append(f"Credits {data['credits']} >= {data['min_credits']} ✅")
            else:
                reasons.append(f"Credits {data['credits']} < {data['min_credits']} ❌")
        
        return {
            "eligible": eligible,
            "answer": "Yes" if eligible else "No",
            "conditions": reasons
        }
    except Exception as e:
        return {"error": str(e), "answer": "Unknown"}


# ============================================================
# MODULE 3B: GIẢI MẠCH ĐIỆN BẰNG SYMPY
# ============================================================
def solve_circuit(data: dict) -> dict:
    """Dùng SymPy giải bài toán mạch điện"""
    
    try:
        V = data.get("voltage", 0)
        resistors = data.get("resistors", [])
        c_type = data.get("circuit_type", "series")
        
        if not resistors or V == 0:
            return {"error": "Thiếu dữ liệu mạch điện"}
        
        if c_type == "series":
            I = symbols('I')
            R_total = sum(resistors)
            pt = Eq(V, I * R_total)
            I_val = float(solve(pt, I)[0])
            voltages = [round(I_val * R, 4) for R in resistors]
            
            return {
                "circuit_type": "series",
                "current_A": round(I_val, 4),
                "R_total_ohm": R_total,
                "voltages_V": voltages,
                "resistors": resistors,
                "source_V": V
            }
        
        else:  # parallel
            currents = [round(V / R, 4) for R in resistors]
            I_total = sum(currents)
            R_eq = round(V / I_total, 4)
            
            return {
                "circuit_type": "parallel",
                "currents_A": currents,
                "I_total_A": round(I_total, 4),
                "R_equivalent_ohm": R_eq,
                "resistors": resistors,
                "source_V": V
            }
    
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# MODULE 4: LLM VIẾT GIẢI THÍCH
# ============================================================
def generate_explanation(question: str, q_type: str, solution: dict) -> str:
    """LLM viết giải thích dựa trên kết quả chính xác từ Z3/SymPy"""
    
    prompt = f"""Question: {question}

Calculated solution (these numbers are 100% correct, use them exactly):
{json.dumps(solution, indent=2)}

Write a clear explanation using ONLY the numbers above.
Format strictly as:
ANSWER: [direct answer]
REASONING: [step by step, reference exact numbers from solution]
"""
    
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor. Use ONLY the provided calculated values. Never recalculate."
            },
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]


# ============================================================
# PIPELINE CHÍNH — KẾT HỢP TẤT CẢ
# ============================================================
def process_question(q: dict) -> dict:
    """Xử lý một câu hỏi từ đầu đến cuối"""
    
    question = q["question"]
    print(f"\n  🔍 Phân loại câu hỏi...")
    
    # Bước 1: Phân loại
    q_type = classify_question(question)
    print(f"  📌 Loại: {q_type.upper()}")
    
    # Bước 2: Trích xuất dữ liệu
    print(f"  📥 Trích xuất dữ liệu...")
    data = extract_data(question, q_type)
    print(f"  📊 Dữ liệu: {data}")
    
    # Bước 3: Giải bằng công cụ phù hợp
    if q_type == "logic":
        print(f"  ⚙️  Z3 đang giải...")
        solution = solve_logic(data)
    else:
        print(f"  ⚙️  SymPy đang giải...")
        solution = solve_circuit(data)
    print(f"  ✅ Kết quả: {solution}")
    
    # Bước 4: LLM viết giải thích
    print(f"  ✍️  LLM đang viết giải thích...")
    explanation = generate_explanation(question, q_type, solution)
    
    return {
        "id": q["id"],
        "question": question,
        "type": q_type,
        "solution": solution,
        "output": explanation
    }


# ============================================================
# CHẠY THỬ VỚI 4 CÂU HỎI MẪU
# ============================================================
test_questions = [
    {
        "id": 1,
        "question": "A student has GPA 3.2 and completed 120 credits. Rules: GPA >= 3.0 AND credits >= 120 to graduate. Is this student eligible?"
    },
    {
        "id": 2,
        "question": "A student has GPA 2.5 and completed 130 credits. Rules: GPA >= 3.0 AND credits >= 120 to graduate. Is this student eligible?"
    },
    {
        "id": 3,
        "question": "R1=10Ω and R2=20Ω are connected in series with a 30V source. Calculate the current and voltage across each resistor."
    },
    {
        "id": 4,
        "question": "R1=10Ω and R2=20Ω are connected in parallel with a 20V source. Calculate the current through each resistor and total current."
    }
]

results = []

print("🚀 BẮT ĐẦU XỬ LÝ HỆ THỐNG EXACT")
print("=" * 55)

for q in test_questions:
    print(f"\n{'='*55}")
    print(f"📝 CÂU {q['id']}: {q['question'][:60]}...")
    print('='*55)
    
    result = process_question(q)
    results.append(result)
    
    print(f"\n{'─'*55}")
    print(result["output"])

# Lưu kết quả ra file
with open("exact_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*55}")
print(f"🎉 HOÀN THÀNH! Đã xử lý {len(results)} câu hỏi.")
print(f"📁 Kết quả lưu tại: exact_results.json")
print(f"{'='*55}")