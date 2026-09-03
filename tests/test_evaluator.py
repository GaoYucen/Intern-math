import math
import unittest

from benchmark.evaluator import extract_final_answer, score_answer


class EvaluatorTest(unittest.TestCase):
    def test_final_marker(self):
        self.assertEqual(extract_final_answer("work\nFINAL_ANSWER: 72"), "72")

    def test_numeric_pi(self):
        s = score_answer("FINAL_ANSWER: 2*pi", "2*pi", "numeric")
        self.assertTrue(s.correct)

    def test_latex_fraction_pi_numeric(self):
        s = score_answer(r"FINAL_ANSWER: \frac{\pi}{6}", str(math.pi / 6), "float")
        self.assertTrue(s.correct)

    def test_latex_unbraced_fraction_numeric(self):
        s = score_answer(r"FINAL_ANSWER: \frac13", "1/3", "rational")
        self.assertTrue(s.correct)

    def test_latex_dfrac_with_implicit_pi_numeric(self):
        s = score_answer(r"FINAL_ANSWER: -\dfrac{4}{25\pi}", str(-4 / (25 * math.pi)), "float")
        self.assertTrue(s.correct)

    def test_latex_inline_wrappers_numeric(self):
        s = score_answer(r"FINAL_ANSWER: \(\frac12\)", "1/2", "rational")
        self.assertTrue(s.correct)

    def test_tex_grouped_integer(self):
        s = score_answer(r"FINAL_ANSWER: 1\,058\,787", "1058787", "integer")
        self.assertTrue(s.correct)

    def test_plain_grouped_integer(self):
        s = score_answer("FINAL_ANSWER: 1,058,787", "1058787", "integer")
        self.assertTrue(s.correct)

    def test_nested_boxed_extraction(self):
        text = r"work $$\boxed{u(x,t)=\frac{1}{1+e^{x-t}}}$$"
        self.assertEqual(extract_final_answer(text), r"u(x,t)=\frac{1}{1+e^{x-t}}")

    def test_unordered_values(self):
        s = score_answer("FINAL_ANSWER: 1, 3", "3,1", "set")
        self.assertTrue(s.correct)

    def test_symbolic_equivalence(self):
        s = score_answer("FINAL_ANSWER: (x+1)^2", "x^2+2*x+1", "symbolic")
        if s.correct is not None:  # sympy may be absent in a bare environment
            self.assertTrue(s.correct)

    def test_symbolic_latex_imaginary_unit(self):
        s = score_answer(r"FINAL_ANSWER: -\pi i", "-I*pi", "symbolic")
        if s.correct is not None:
            self.assertTrue(s.correct)

    def test_symbolic_latex_log(self):
        s = score_answer(r"FINAL_ANSWER: \ln 4-\frac54", "-5/4 + log(4)", "symbolic")
        if s.correct is not None:
            self.assertTrue(s.correct)

    def test_symbolic_plain_ln(self):
        s = score_answer("FINAL_ANSWER: 2 ln 2 - 5/4", "log(4)-5/4", "symbolic")
        if s.correct is not None:
            self.assertTrue(s.correct)

    def test_symbolic_pi_power(self):
        s = score_answer(r"FINAL_ANSWER: \frac{\pi^6}{504}", "pi^6/504", "symbolic")
        if s.correct is not None:
            self.assertTrue(s.correct)

    def test_symbolic_latex_containment(self):
        s = score_answer(
            r"FINAL_ANSWER: The submanifolds are precisely $\mathbb{R}P^k$.",
            r"\mathbb{R}P^k",
            "symbolic",
        )
        self.assertTrue(s.correct)

    def test_boolean_prose(self):
        s = score_answer(
            "work\nFINAL_ANSWER: No, we cannot reject H_0.",
            False,
            "boolean",
        )
        self.assertTrue(s.correct)

    def test_boolean_sentence_true(self):
        s = score_answer(
            "FINAL_ANSWER: The stated equation is true for the manifold.",
            True,
            "boolean",
        )
        self.assertTrue(s.correct)

    def test_truncated_choice_is_not_scored_from_incidental_letter(self):
        s = score_answer(
            "Long unfinished derivation ending with: But a(k) is usually small",
            "C",
            "choice",
        )
        self.assertIsNone(s.correct)
        self.assertEqual(s.reason, "missing explicit final choice")

    def test_strict_choice_fallback(self):
        s = score_answer("B", "B", "choice")
        self.assertTrue(s.correct)

    def test_proof_requires_review(self):
        s = score_answer("FINAL_ANSWER: proved", "proof", "proof")
        self.assertIsNone(s.correct)


if __name__ == "__main__":
    unittest.main()
