"""CPU checks for the M1 x T2 combination replay job: subset choice and the summary arithmetic."""

from pathlib import Path
import tomllib
import unittest

from training.kaggle.m1_combo import ARMS, arm_summary, select_subset

JOB = Path(__file__).resolve().parents[1] / "training/jobs/p2-m1-combo.toml"


class ComboTests(unittest.TestCase):
    def test_subset_takes_the_first_prompts_of_each_task(self):
        validation = [dict(task=task, n=n) for task in ("math", "code", "instruction", "summarization")
                      for n in range(80)]
        rows = select_subset(validation, 20)
        self.assertEqual(len(rows), 80)
        self.assertEqual([row["n"] for row in rows], list(range(20)) * 4)
        with self.assertRaisesRegex(ValueError, "fewer than 81"):
            select_subset(validation, 81)

    def test_summary_counts_and_rates(self):
        prompt = dict(accepted=6, proposed=9, rounds=4, generated_tokens=10, draft_steps=9, cache_rounds=1)
        per_prompt = [dict(prompt, task=task) for task in ("math", "code", "instruction", "summarization")]
        summary = arm_summary(per_prompt, 0.0623)
        self.assertEqual((summary["accepted"], summary["proposed"], summary["prompts"]), (24, 36, 4))
        self.assertAlmostEqual(summary["acceptance"], 2 / 3)
        self.assertAlmostEqual(summary["tokens_per_round"], 2.5)
        self.assertAlmostEqual(summary["modeled_cost_per_token"], (16 + 0.0623 * 36) / 40)
        self.assertAlmostEqual(summary["cache_round_share"], 0.25)
        self.assertAlmostEqual(summary["per_task"]["code"]["acceptance"], 2 / 3)

    def test_job_runs_four_arms_on_the_recorded_t2_weights(self):
        job = tomllib.loads(JOB.read_text())
        self.assertEqual(ARMS, (("B1", "baseline"), ("B1", "m1"), ("T2", "baseline"), ("T2", "m1")))
        self.assertEqual((job["pair"], job["k"], job["prompts_per_task"]), ("P2", 3, 20))
        self.assertRegex(job["t2_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(0 < job["preflight_minutes"] <= 5 < job["max_minutes"])


if __name__ == "__main__":
    unittest.main()
