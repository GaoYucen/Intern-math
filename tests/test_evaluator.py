import unittest

from benchmark.evaluator import extract_final_answer, score_answer


class EvaluatorTest(unittest.TestCase):
    def test_final_marker(self):
        self.assertEqual(extract_final_answer("work\nFINAL_ANSWER: 72"), "72")

    def test_numeric_pi(self):
        s = score_answer("FINAL_ANSWER: 2*pi", "2*pi", "numeric")
        self.assertTrue(s.correct)

    def test_unordered_values(self):
        s = score_answer("FINAL_ANSWER: 1, 3", "3,1", "set")
        self.assertTrue(s.correct)

    def test_symbolic_equivalence(self):
        s = score_answer("FINAL_ANSWER: (x+1)^2", "x^2+2*x+1", "symbolic")
        if s.correct is not None:  # sympy may be absent in a bare environment
            self.assertTrue(s.correct)

    def test_proof_requires_review(self):
        s = score_answer("FINAL_ANSWER: proved", "proof", "proof")
        self.assertIsNone(s.correct)


if __name__ == "__main__":
    unittest.main()
