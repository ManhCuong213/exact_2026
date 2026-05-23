# EXACT 2026 — Hệ thống AI Giáo dục

## Cấu trúc thư mục
```
exact2026/
├── main.py                  ← Pipeline chính (Bạn - PM)
├── modules/
│   ├── classifier.py        ← Member 2
│   ├── logic_solver.py      ← Member 3
│   ├── physics_solver.py    ← Member 4
│   └── explainer.py         ← Member 5
├── data/
│   ├── Logic_Based_Educational_Queries.json
│   └── Physics_Problems_Text_Only.csv
├── tests/
│   ├── test_classifier.py
│   ├── test_logic.py
│   ├── test_physics.py
│   └── test_explainer.py
├── results/
│   └── output.json          ← Kết quả nộp bài
└── README.md

## Cài đặt môi trường
```
py -m venv venv
venv\Scripts\activate
py -m pip install ollama z3-solver sympy
```

## Chạy hệ thống
```
py main.py
```

## Quy tắc nhóm
1. Mỗi người làm trên branch riêng
2. Họp 2 lần/tuần (30 phút)
3. Deadline nội bộ: cuối Tuần 2 phải có code chạy được
4. Nếu kẹt quá 2 tiếng → hỏi nhóm ngay
