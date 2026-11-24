"""CPU checks for the training red-flag CLI, on synthetic heartbeat lines in the trainer's format."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import tomllib
import unittest

try:
    import matplotlib
except ImportError:
    matplotlib = None

from training.training_red_flag import GREEN, NEUTRAL, RED, evaluate, parse_beats, report, verdict

JOBS = Path(__file__).resolve().parents[1] / "training/jobs"


def job(name):
    return tomllib.loads((JOBS / f"{name}.toml").read_text())


def beat(clock, text):
    return f"[heartbeat {clock}Z] {text}"


def result(tier, acceptance, gain):
    return beat("06:30:00", f"phase=result tier={tier} acceptance={acceptance} b1=0.1863 gain_pp={gain} "
                            "math=0.4 code=0.3 instruction=0.3 summarization=0.35")


HEALTHY = [
    beat("05:40:00", "phase=preflight action=check_inputs"),
    beat("05:42:00", "phase=B1-replay step=0/320"),
    beat("05:44:00", "phase=B1-replay step=60/320 eta=5m acceptance=0.2150"),
    beat("05:49:00", "phase=B1-replay step=320/320 eta=0m00s acceptance=0.1863"),
    beat("05:49:01", "phase=T2-fit-1 action=load_drafter"),
    beat("05:49:02", "phase=T2-fit-1 step=0/360"),
    beat("05:49:32", "phase=T2-fit-1 step=120/360 eta=1m loss=2.817"),
    beat("05:50:30", "phase=T2-fit-1 step=360/360 eta=0m00s loss=2.249"),
    beat("05:50:31", "phase=T2-score-1-epoch-2 step=320/320 eta=0m00s top1_agreement=0.5049"),
    beat("05:51:00", "phase=T2-score-1-epoch-4 step=320/320 eta=0m00s top1_agreement=0.5186"),
]


def flags(checks):
    return {check.name: check.flag for check in checks}


class ParseTests(unittest.TestCase):
    def test_fields_steps_and_values_with_spaces(self):
        beats = parse_beats(["noise", beat("23:59:50", "phase=T2-fit-2 step=7/90 eta=1m loss=2.5"),
                             beat("00:00:10", "phase=abort reason=loss is nan in phase T2-fit-2")])
        self.assertEqual([(b.phase, b.step, b.total) for b in beats], [("T2-fit-2", 7, 90), ("abort", None, None)])
        self.assertEqual(beats[0].fields, {"eta": "1m", "loss": "2.5"})
        self.assertEqual(beats[1].fields["reason"], "loss is nan in phase T2-fit-2")
        self.assertEqual(beats[1].clock - beats[0].clock, 20, "Crossing midnight keeps the clock increasing.")


class FlagTests(unittest.TestCase):
    def test_healthy_full_run_is_green_and_unfinished_steps_wait(self):
        lines = HEALTHY + [beat("05:51:30", "phase=T2-fit-2 step=0/360")]
        now = datetime(2026, 11, 24, 5, 52, 0, tzinfo=timezone.utc)
        checks, now_line = evaluate(parse_beats(lines), job("p2"), status="running", now=now)
        self.assertEqual(verdict(checks), GREEN)
        self.assertEqual(flags(checks)["B1 replay"], GREEN)
        self.assertEqual(flags(checks)["T2 fit 1"], GREEN)
        self.assertIn("rank 16, lr 0.0001", next(c.detail for c in checks if c.name == "T2 fit 1"))
        self.assertTrue(next(c for c in checks if c.name == "T2 fit 2").pending)
        self.assertTrue(now_line.startswith("T2-fit-2 step 0/360"))
        text, code = report("title", now_line, checks)
        self.assertIn("Overall: GREEN-FLAG", text)
        self.assertEqual(code, 0)

    def test_b1_far_from_expected_is_red(self):
        lines = [beat("05:49:00", "phase=B1-replay step=320/320 acceptance=0.1100")]
        checks, _ = evaluate(parse_beats(lines), job("p2"))
        self.assertEqual(flags(checks)["B1 replay"], RED)
        self.assertEqual(report("t", "", checks)[1], 2)

    def test_abort_and_traceback_are_red(self):
        abort = [beat("05:49:00", "phase=abort reason=Job exceeded max_minutes.")]
        checks, _ = evaluate(parse_beats(abort), job("p2"), status="error")
        self.assertEqual(flags(checks)["abort"], RED)
        checks, _ = evaluate([], job("p2"), errors=["Traceback (most recent call last):"])
        self.assertEqual(verdict(checks), RED)

    def test_loss_rising_is_red_and_flat_is_neutral(self):
        for end, expected in (("3.3", RED), ("2.80", NEUTRAL)):
            lines = [beat("05:49:32", "phase=T2-fit-1 step=10/360 loss=2.8"),
                     beat("05:50:02", f"phase=T2-fit-1 step=40/360 loss={end}")]
            checks, _ = evaluate(parse_beats(lines), job("p2"))
            self.assertEqual(flags(checks)["T2 fit 1"], expected, end)

    def test_finished_fit_with_flat_loss_defers_to_a_green_score(self):
        lines = [beat("06:10:00", "phase=T1-fit-4 step=90/182 loss=2.545"),
                 beat("06:10:30", "phase=T1-fit-4 step=182/182 loss=2.750"),
                 beat("06:10:40", "phase=T1-score-4-epoch-2 step=320/320 top1_agreement=0.4546")]
        checks, _ = evaluate(parse_beats(lines), job("p2"))
        self.assertEqual(flags(checks)["T1 fit 4"], GREEN)
        unscored = evaluate(parse_beats(lines[:2]), job("p2"))[0]
        self.assertEqual(flags(unscored)["T1 fit 4"], NEUTRAL)

    def test_top1_dropping_between_checkpoints_is_neutral(self):
        lines = [beat("05:50:31", "phase=T2-score-3-epoch-2 step=320/320 top1_agreement=0.60"),
                 beat("05:51:00", "phase=T2-score-3-epoch-4 step=320/320 top1_agreement=0.58")]
        checks, _ = evaluate(parse_beats(lines), job("p2"))
        self.assertEqual(flags(checks)["T2 score 3 ep 4"], NEUTRAL)
        below = [beat("05:50:31", "phase=T2-score-1-epoch-2 step=320/320 top1_agreement=0.39")]
        self.assertEqual(flags(evaluate(parse_beats(below), job("p2"))[0])["T2 score 1 ep 2"], RED)

    def test_result_gain_against_the_two_point_bar(self):
        for gain, expected in (("6.4", GREEN), ("1.0", NEUTRAL), ("-0.5", RED)):
            checks, _ = evaluate(parse_beats([result("T2", 0.25, gain)]), job("p2"))
            self.assertEqual(flags(checks)["T2 result"], expected, gain)
        checks, _ = evaluate(parse_beats([result("T1", 0.18, "-0.5")]), job("p2"))
        self.assertEqual(flags(checks)["T1 result"], NEUTRAL, "T1 below B1 does not block the plan.")
        checks, _ = evaluate(parse_beats([result("T2", 0.19, "0.1")]), job("p2-smoke"))
        self.assertEqual(flags(checks)["T2 result"], GREEN, "The smoke checks plumbing, not gain.")

    def test_running_replay_is_compared_with_b1_at_the_same_prompt(self):
        lines = HEALTHY + [beat("05:55:00", "phase=T2-replay step=60/320 acceptance=0.4043")]
        checks, _ = evaluate(parse_beats(lines), job("p2"))
        replay = next(c for c in checks if c.name == "T2 replay")
        self.assertEqual((replay.flag, replay.pending), (GREEN, False))
        self.assertIn("B1 was 0.2150 at prompt 60", replay.detail)

    def test_silence_budget_and_missing_results(self):
        lines = [beat("05:40:00", "phase=preflight action=check_inputs"),
                 beat("07:30:00", "phase=T1-fit-4 step=10/200 loss=3.0")]
        late = datetime(2026, 11, 24, 7, 41, 0, tzinfo=timezone.utc)
        checks, _ = evaluate(parse_beats(lines), job("p2"), status="running", now=late)
        self.assertEqual(flags(checks)["heartbeat"], RED)
        self.assertEqual(flags(checks)["time"], RED)
        checks, _ = evaluate(parse_beats([result("T2", 0.25, "6.4")]), job("p2"), status="complete")
        self.assertEqual(flags(checks)["results"], RED, "The full run also owes a T1 result.")
        checks, _ = evaluate([], job("p2"), status="complete")
        self.assertEqual((verdict(checks), flags(checks)["log"]), (NEUTRAL, NEUTRAL),
                         "A finished job whose live log is gone needs collecting, not a red flag.")
        checks, _ = evaluate([], job("p2"), status="queued")
        self.assertEqual((verdict(checks), checks[0].pending), (NEUTRAL, True))

    def test_replay_only_arms_are_named_apart_and_never_crash(self):
        lines = [beat("06:00:00", "phase=B1-baseline-replay step=80/80 acceptance=0.19"),
                 beat("06:00:01", "phase=result arm=B1-baseline acceptance=0.19 tokens_per_round=1.5 cost=0.7"),
                 beat("06:02:00", "phase=B1-m1-replay step=80/80 acceptance=0.30"),
                 beat("06:02:01", "phase=result arm=B1-m1 acceptance=0.30 tokens_per_round=1.8 cost=0.6")]
        checks, _ = evaluate(parse_beats(lines), {"pair": "P2", "max_minutes": 45})
        self.assertEqual(flags(checks)["B1 replay"], GREEN)
        self.assertEqual(flags(checks)["B1 m1 replay"], GREEN, "M1 is not judged against baseline B1.")
        self.assertEqual(flags(checks)["B1-m1 result"], GREEN)


@unittest.skipIf(matplotlib is None, "matplotlib is required for --visual")
class FigureTests(unittest.TestCase):
    def test_visual_draws_every_panel_from_heartbeats(self):
        from training.red_flag_figure import draw

        lines = HEALTHY + [beat("05:55:00", "phase=T2-replay step=60/320 acceptance=0.4043"),
                           result("T2", 0.3146, "12.83")]
        beats = parse_beats(lines)
        checks, now_line = evaluate(beats, job("p2"))
        with tempfile.TemporaryDirectory() as folder:
            path = draw(Path(folder) / "run.png", "title", now_line, checks, verdict(checks), beats, job("p2"))
            self.assertGreater(path.stat().st_size, 20000)
            self.assertTrue(path.read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a")))


if __name__ == "__main__":
    unittest.main()
