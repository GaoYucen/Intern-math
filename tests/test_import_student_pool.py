import unittest

from scripts.import_student_pool import normalize_answer_type, normalize_row


class ImportStudentPoolTest(unittest.TestCase):
    def test_free_form_integer_is_inferred_conservatively(self):
        self.assertEqual(normalize_answer_type("free_form", "72"), "integer")

    def test_unknown_free_form_stays_text(self):
        self.assertEqual(normalize_answer_type("free_form", "x\\in\\mathbb{R}"), "text")

    def test_mapped_subject_is_normalized(self):
        row = {
            "problem": "Compute 1+1",
            "answer": "2",
            "answer_type": "integer",
            "subject": "高等代数",
            "data_status": "mapped",
            "source_dataset": "demo",
        }
        out = normalize_row(row)
        self.assertIsNotNone(out)
        self.assertEqual(out["domain"], "advanced_algebra")
        self.assertEqual(out["benchmark_status"], "candidate")

    def test_unmapped_rows_are_rejected(self):
        self.assertIsNone(normalize_row({"data_status": "unmapped"}))


if __name__ == "__main__":
    unittest.main()
