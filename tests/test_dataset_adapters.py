from benchmark.adapters import (
    canonical_answer_type,
    infer_domain,
    normalize_scibench,
    normalize_supergpqa,
    normalize_theoremqa,
)


def test_theoremqa_math_row_normalizes():
    row = normalize_theoremqa({
        "Question": "Compute the contour integral.",
        "Answer": -1.047,
        "Answer_type": "float",
        "Picture": None,
        "source": "example",
        "id": "x/1",
        "explanation": "NONE",
        "theorem": "residue theorem",
        "subfield": "Complex analysis",
        "field": "Math",
    })
    assert row is not None
    assert row["domain"] == "complex_analysis"
    assert row["answer_type"] == "float"
    assert row["problem"] == "Compute the contour integral."
    assert row["review_status"] == "pending"
    assert len(row["problem_hash"]) == 64


def test_scibench_diff_maps_to_ode():
    row = normalize_scibench({
        "problem_text": "Solve y' = y.",
        "answer_number": "2.71828",
        "answer_latex": "e",
        "unit": "",
        "problemid": "1",
    }, "diff")
    assert row is not None
    assert row["domain"] == "ode"
    assert row["answer_type"] == "numeric"


def test_supergpqa_math_choice_formats_options():
    row = normalize_supergpqa({
        "uuid": "abc",
        "question": "Which statement is true?",
        "options": ["first", "second", "third", "fourth"],
        "answer_letter": "B",
        "answer": "second",
        "discipline": "Mathematics",
        "field": "Mathematical Analysis",
        "subfield": "Real Analysis",
        "difficulty": "hard",
        "is_calculation": False,
    })
    assert row is not None
    assert row["domain"] == "mathematical_analysis"
    assert row["answer"] == "B"
    assert row["answer_type"] == "choice"
    assert "A. first" in row["problem"]
    assert "B. second" in row["problem"]


def test_non_math_supergpqa_is_dropped():
    assert normalize_supergpqa({
        "discipline": "Engineering",
        "field": "Circuits",
        "subfield": "Circuits",
        "question": "x",
        "answer_letter": "A",
        "options": ["a", "b"],
    }) is None


def test_domain_rules_prioritize_pde():
    assert infer_domain("Partial Differential Equations", "analysis") == "pde"


def test_list_answer_type():
    assert canonical_answer_type("list of integer", [1, 2]) == "multiple_values"
