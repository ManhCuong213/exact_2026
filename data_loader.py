"""
Data Loader - Load & Search Logic and Physics Dataset
"""

import json
import pandas as pd
import os
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

# Cache datasets
_LOGIC_DATA = None
_PHYSICS_DATA = None


def load_logic_data() -> List[Dict[str, Any]]:
    """Load Logic-Based Educational Queries from JSON"""
    global _LOGIC_DATA
    if _LOGIC_DATA is not None:
        return _LOGIC_DATA
    
    try:
        with open("data/Logic_Based_Educational_Queries.json", "r", encoding="utf-8") as f:
            _LOGIC_DATA = json.load(f)
        for entry in _LOGIC_DATA:
            entry.pop("explanation", None)
        print(f"✓ Loaded {len(_LOGIC_DATA)} logic entries (explanation field stripped)")
        return _LOGIC_DATA
    except Exception as e:
        print(f"✗ Error loading logic data: {e}")
        return []


def load_physics_data() -> pd.DataFrame:
    """Load Physics Problems from CSV"""
    global _PHYSICS_DATA
    if _PHYSICS_DATA is not None:
        return _PHYSICS_DATA
    
    try:
        _PHYSICS_DATA = pd.read_csv("data/Physics_Problems_Text_Only.csv")
        print(f"✓ Loaded {len(_PHYSICS_DATA)} physics problems")
        return _PHYSICS_DATA
    except Exception as e:
        print(f"✗ Error loading physics data: {e}")
        return pd.DataFrame()


def _similarity_score(s1: str, s2: str) -> float:
    """Calculate similarity between two strings (0-1)"""
    s1 = str(s1).lower().strip()
    s2 = str(s2).lower().strip()
    return SequenceMatcher(None, s1, s2).ratio()


def search_similar_logic_questions(
    question: str, 
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search similar logic questions in dataset
    
    Returns list of similar entries with:
    - similarity_score
    - idx, premises-NL, questions, answers
    """
    logic_data = load_logic_data()
    if not logic_data:
        return []
    
    results = []
    
    for entry in logic_data:
        # Score against all questions in entry
        best_score = 0
        for q in entry.get("questions", []):
            score = _similarity_score(question, q)
            best_score = max(best_score, score)
        
        if best_score > 0.3:  # Threshold
            results.append({
                "similarity_score": round(best_score, 3),
                "idx": entry.get("idx"),
                "premises_nl": entry.get("premises-NL", [])[:3],  # Top 3 premises
                "premises_fol": entry.get("premises-FOL", [])[:3],  # Top 3 FOL
                "questions": entry.get("questions", []),
                "answers": entry.get("answers", []),
            })
    
    # Sort by similarity and return top K
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]


def search_similar_physics_problems(
    question: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search similar physics problems in dataset
    
    Returns list of similar problems with:
    - similarity_score
    - question, answer, unit, solution_steps
    """
    physics_data = load_physics_data()
    if physics_data.empty:
        return []
    
    results = []
    
    for idx, row in physics_data.iterrows():
        problem_text = str(row.get("question", "")) if "question" in row.index else ""
        score = _similarity_score(question, problem_text)
        
        if score > 0.3:  # Threshold
            result = {
                "similarity_score": round(score, 3),
                "id": idx,
            }
            
            # Add all available columns from CSV
            for col in row.index:
                if pd.notna(row[col]):
                    result[col] = str(row[col])
            
            results.append(result)
    
    # Sort by similarity and return top K
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]


def get_dataset_stats() -> Dict[str, Any]:
    """Get statistics about datasets"""
    logic_data = load_logic_data()
    physics_data = load_physics_data()
    
    return {
        "logic": {
            "total_entries": len(logic_data),
            "total_questions": sum(len(entry.get("questions", [])) for entry in logic_data),
        },
        "physics": {
            "total_problems": len(physics_data),
            "columns": list(physics_data.columns) if not physics_data.empty else [],
        }
    }
