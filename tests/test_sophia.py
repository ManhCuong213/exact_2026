import sys
import os

# Thêm thư mục gốc project (cha của tests/) vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.logic_solver import solve_logic
from modules.explainer import generate_explanation

# Khai báo sẵn câu hỏi của Sophia
pq = {
    "question": "Based on the above premises, which is the strongest conclusion?\nA. Sophia qualifies for the university scholarship\nB. Sophia needs a faculty recommendation\nC. Sophia is eligible for the international program\nD. Sophia needs to pass the language proficiency exam",
    "type": "mcq",
    "premises": [
        "Sophia has completed the core curriculum.",
        "Sophia has passed the science assessment.",
        "Sophia has completed the research methodology course.",
        "Sophia has completed her capstone project.",
        "Sophia has completed the required community service hours.",
        "Students who complete core curriculum and pass science assessment qualify for advanced courses.",
        "Students who qualify for advanced courses and complete research methodology are eligible for international program.",
        "Students eligible for international program who complete capstone project are awarded honors diploma.",
        "Students awarded honors diploma who complete community service qualify for scholarship."
    ]
}

print("🧠 Hệ thống đang bốc câu hỏi Sophia và giải toán...")

# Gọi Z3 Solver và Qwen 2.5 giải bài
sol = solve_logic(pq)
exp = generate_explanation(pq["question"], "logic", sol, premises=pq["premises"])

print(f"\n🎯 ĐÁP ÁN AI: {sol.get('answer', 'Không rõ')}")
print(f"\n📝 LỜI GIẢI CHI TIẾT:\n{exp}")
