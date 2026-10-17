"""Final report reproducibility and rejection of misleading benchmark inputs."""

from copy import deepcopy
import json
import unittest

from scripts.build_report import ROOT, load_reports, metric_table, render_report, verify_result


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = load_reports()
        cls.manifest = json.loads((ROOT / "datasets/online-shoppers.json").read_text())

    def test_report_matches_verified_artifacts_without_refitting(self):
        content = render_report(self.reports)
        self.assertEqual(content, (ROOT / "results/benchmark_results.md").read_text())
        self.assertEqual(content.count("Evaluated: 7/7"), 3)
        for report in self.reports:
            self.assertIn(report["sha256"], content)
            self.assertIn(metric_table(report["result"]), content)

    def test_smoke_partial_and_inconsistent_scores_are_rejected(self):
        for mutation, message in (("profile", "full public-dataset"),
                                  ("rows", "every source row"),
                                  ("metric", "disagree with saved predictions")):
            with self.subTest(mutation=mutation):
                result = deepcopy(self.reports[0]["result"])
                if mutation == "profile":
                    result["dataset"]["profile"] = "smoke"
                elif mutation == "rows":
                    result["dataset"]["source_row_positions"].pop()
                else:
                    result["models"][0]["metrics"]["test"]["roc_auc"] = 0.99
                with self.assertRaisesRegex(ValueError, message):
                    verify_result(result, self.manifest, "binary")

    def test_unavailable_and_failed_rows_keep_reasons_without_substitute_scores(self):
        for status, label in (("unavailable", "Not evaluated"), ("failed", "Failed")):
            with self.subTest(status=status):
                reports = deepcopy(self.reports)
                row = reports[0]["result"]["models"][-1]
                row.update(status=status, reason="Recorded test runtime limitation", metrics={}, predictions={})
                verify_result(reports[0]["result"], self.manifest, "binary")
                content = render_report(reports)
                self.assertIn("Evaluated: 6/7", content)
                self.assertIn(f"**Mitra v2 - Engineered: {label}** — Recorded test runtime limitation", content)
                self.assertIn(f"| Mitra v2 - Engineered | {label} |", content)
                row["metrics"] = {"test": {"roc_auc": 0.99}}
                with self.assertRaisesRegex(ValueError, "must not contain scores"):
                    verify_result(reports[0]["result"], self.manifest, "binary")


if __name__ == "__main__":
    unittest.main()
