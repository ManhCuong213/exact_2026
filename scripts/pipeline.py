import ollama
import json

# ---- Hàm hỏi LLM ----
def ask_llm(question: str) -> str:
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert reasoning assistant. "
                    "Always structure your response as:\n"
                    "ANSWER: [Yes/No]\n"
                    "REASONING: [step-by-step explanation]"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    return response["message"]["content"]

# ---- Đọc file câu hỏi ----
with open("questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# ---- Xử lý từng câu ----
results = []

for q in questions:
    print(f"\n📝 Đang xử lý câu {q['id']}...")
    
    answer = ask_llm(q["question"])
    
    result = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "output": answer
    }
    results.append(result)
    print(f"✅ Câu {q['id']} xong!")

# ---- Xuất kết quả ra file ----
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n🎉 Hoàn thành! Kết quả lưu trong file results.json")
print(f"Đã xử lý {len(results)} câu hỏi.")