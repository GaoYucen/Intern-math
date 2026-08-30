import unittest

from scripts.adjudicate_results import parse_judge_response


class JudgeParserTest(unittest.TestCase):
    def assert_verdict(self, text, expected):
        verdict, _reason, parsed, _method = parse_judge_response(text)
        self.assertTrue(parsed)
        self.assertEqual(verdict, expected)

    def test_preferred_verdict_line(self):
        self.assert_verdict(
            "VERDICT: CORRECT\nREASON: Equivalent to the reference.",
            "correct",
        )

    def test_markdown_verdict_line(self):
        self.assert_verdict(
            "**VERDICT**: WRONG\n**REASON**: Sign error.",
            "wrong",
        )

    def test_json_inside_markdown(self):
        self.assert_verdict(
            'Here is the result:\n```json\n{"verdict":"invalid","reason":"broken gold"}\n```',
            "invalid",
        )

    def test_json_with_braces_in_reason(self):
        self.assert_verdict(
            '{"verdict":"correct","reason":"The set {1,2} matches."}',
            "correct",
        )

    def test_standalone_heading(self):
        self.assert_verdict("Analysis omitted.\n**UNCERTAIN**", "uncertain")

    def test_alias_incorrect(self):
        self.assert_verdict("VERDICT: INCORRECT", "wrong")

    def test_unparseable_is_not_silently_classified(self):
        verdict, reason, parsed, method = parse_judge_response(
            "The response discusses several possible approaches but never gives a decision."
        )
        self.assertFalse(parsed)
        self.assertEqual(verdict, "uncertain")
        self.assertEqual(method, "unparsed")
        self.assertIn("could not be parsed", reason)


if __name__ == "__main__":
    unittest.main()
