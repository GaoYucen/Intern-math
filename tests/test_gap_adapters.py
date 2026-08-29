from benchmark.gap_adapters import (
    deepmath_gap_domain,
    normalize_deepmath_gap,
    normalize_orqa,
)


def test_orqa_zero_based_target_to_letter():
    row = normalize_orqa({
        "QUESTION_TYPE": "Q6",
        "CONTEXT": "Choose which jobs to schedule to maximize reward.",
        "QUESTION": "What is the decision variable?",
        "OPTIONS": ["deadline", "job selection", "processing time", "reward"],
        "TARGET_ANSWER": 1,
        "REASONING": "The decision is whether to select a job.",
    }, "validation:0")
    assert row is not None
    assert row["domain"] == "operations_research"
    assert row["answer_type"] == "choice"
    assert row["answer"] == "B"
    assert "Question:" in row["problem"]
    assert "B. job selection" in row["problem"]


def test_deepmath_pde_gap_row():
    row = normalize_deepmath_gap({
        "question": "Solve the heat equation under the stated boundary conditions.",
        "final_answer": "u(x,t)=e^{-t}\\sin x",
        "difficulty": 14,
        "topic": "Mathematics -> Differential Equations -> Partial Differential Equations (PDEs)",
        "failed_count": 12,
        "processing_success": True,
    }, "train:10")
    assert row is not None
    assert row["domain"] == "pde"
    assert row["answer_type"] == "symbolic"
    assert row["review_status"] == "pending"


def test_deepmath_differential_geometry_mapping():
    assert deepmath_gap_domain(
        "Mathematics -> Geometry -> Differential Geometry -> Manifolds"
    ) == "differential_geometry"


def test_deepmath_ignores_already_well_covered_topics():
    assert normalize_deepmath_gap({
        "question": "Compute 1+1.",
        "final_answer": "2",
        "difficulty": 1,
        "topic": "Mathematics -> Algebra -> Elementary Algebra",
        "processing_success": True,
    }, "train:11") is None
