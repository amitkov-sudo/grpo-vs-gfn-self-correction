"""Checks of scripts/summarize_runs.py on made-up run folders."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("summarize_runs", os.path.join(ROOT, "scripts", "summarize_runs.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def make_run(out, name, condition, n_steps, evals, done, t_step=100.0):
    d = os.path.join(out, name)
    os.makedirs(d)
    json.dump({"condition": condition, "steps": 60}, open(os.path.join(d, "config.json"), "w"))
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        for e in evals:
            if e["step"] == 0:
                f.write(json.dumps({"eval": e}) + "\n")
        for s in range(1, n_steps + 1):
            f.write(json.dumps({"step": s, "log_ratio": 0.5 * s, "t_step": t_step}) + "\n")
            for e in evals:
                if e["step"] == s:
                    f.write(json.dumps({"eval": e}) + "\n")
    if done:
        os.makedirs(os.path.join(d, "adapter"))


def fix_eval(step, fixed, alarms, t_eval=50.0):
    return {"step": step, "fix_rate": fixed / 60, "fix_rate_count": f"{fixed}/60", "clean_right_count": "55/60",
            "false_alarms": alarms, "fix_pass_at_8": 0.5, "fix_vote_at_8": 0.4, "fix_rate_sampled": 0.25,
            "false_alarm_rate_sampled": 0.1, "t_eval": t_eval}


def test_tables_and_time_left(tmp_path, capsys):
    out = str(tmp_path)
    evals = [fix_eval(s, 4 + s // 20, 1) for s in (0, 20, 40, 60)]
    make_run(out, "phase1-T5-s0", "T5", 60, evals, done=True)
    make_run(out, "phase1-T4-s0", "T4", 30, evals[:2], done=False)
    S.main(["--out", out, "--prefix", "phase1-", "--steps", "60", "--eval-every", "20",
            "--expect", "phase1-T5-s0", "phase1-T4-s0", "phase1-T3-s0"])
    text = capsys.readouterr().out
    assert "| phase1-T5-s0 | T5 | 60 | 7/60 | 1/60 | 55/60 | 50.0% | 40.0% | 25.0% | 10.0% |" in text
    assert "| phase1-T4-s0 | T4 | 30/60 | 15.00 |  | 100 s | no |" in text
    assert "No location test yet." in text
    assert "No approval counts yet." in text
    assert "Queue: 1 of 3 runs finished." in text
    # T4 has 30 steps and 2 evaluations left, T3 all 60 steps and 4 evaluations: 90 * 100 s + 6 * 50 s = 9,300 s
    assert "about 2 h 35 min" in text


def test_a_rerun_with_a_timestamp_counts_as_the_queue_entry(tmp_path):
    out = str(tmp_path)
    make_run(out, "phase1-T5-s0", "T5", 10, [fix_eval(0, 4, 1)], done=False)
    make_run(out, "phase1-T5-s0-20261118-101500", "T5", 60, [fix_eval(0, 4, 1)], done=True)
    runs = S.find_runs(out, "phase1-")
    assert S.match(runs, "phase1-T5-s0")["done"]
    assert S.time_left(runs, ["phase1-T5-s0"], 60, 20) == 0.0


def test_location_table_lists_each_step(tmp_path, capsys):
    out = str(tmp_path)
    final = fix_eval(60, 9, 2)
    for L, fixed in ((2, 30), (3, 20), (4, 10)):
        final.update({f"loc{L}_fix_count": f"{fixed}/50", f"loc{L}_fix_vote_at_8": fixed / 50 + 0.1,
                      f"loc{L}_false_alarms": "2/50"})
    make_run(out, "poison-TB-p25-s0", "TB, own attempts, poison 25%", 60, [fix_eval(0, 4, 1), final], done=True)
    S.main(["--out", out, "--prefix", "poison-"])
    text = capsys.readouterr().out
    assert ("| step 2: fixed | step 2: vote@8 | step 2: false alarms | step 3: fixed | step 3: vote@8 "
            "| step 3: false alarms | step 4: fixed | step 4: vote@8 | step 4: false alarms |") in text
    assert "| poison-TB-p25-s0 | TB, own attempts, poison 25% | 60 | 30/50 | 70.0% | 2/50 | 20/50 | 50.0% | 2/50 | 10/50 | 30.0% | 2/50 |" in text


def test_approval_table(tmp_path, capsys):
    out = str(tmp_path)
    e = fix_eval(60, 9, 2)
    e.update({"approves_on_wrong": 12, "approves_on_clean": 3})
    make_run(out, "attack-TB-a25-s0", "TB, own attempts, attack 25%", 60, [fix_eval(0, 4, 1), e], done=True)
    S.main(["--out", out, "--prefix", "attack-"])
    assert "| attack-TB-a25-s0 | TB, own attempts, attack 25% | 60 | 12/60 | 3/60 |" in capsys.readouterr().out


def test_a_run_stopped_by_its_time_limit(tmp_path, capsys):
    out = str(tmp_path)
    make_run(out, "prelim-T5-s0", "T5", 41, [fix_eval(0, 4, 1), fix_eval(20, 6, 1), fix_eval(41, 8, 2)], done=True)
    with open(os.path.join(out, "prelim-T5-s0", "log.jsonl"), "a") as f:
        f.write(json.dumps({"stopped_early": {"after_step": 41, "max_hours": 10.5, "hours": 10.4}}) + "\n")
    S.main(["--out", out, "--prefix", "prelim-"])
    text = capsys.readouterr().out
    assert "| prelim-T5-s0 | T5 | 41/60 | 20.50 |  | 100 s | stopped after step 41 (time limit) |" in text
    assert "Hit the time limit and stopped early: prelim-T5-s0." in text
    assert "| prelim-T5-s0 | T5 | 41 | 8/60 | 2/60 | 55/60 |" in text


def test_reward_check_table_for_subtb_runs(tmp_path, capsys):
    out = str(tmp_path)
    make_run(out, "prelim-SubTB-wait-s0", "SubTB, Wait answers", 0, [fix_eval(0, 1, 0)], done=False)
    with open(os.path.join(out, "prelim-SubTB-wait-s0", "log.jsonl"), "a") as f:
        for s, (auc_y, auc_r) in enumerate([(0.9, 0.6), (0.7, 0.4), (None, None)], start=1):
            f.write(json.dumps({"step": s, "log_ratio": 0.0, "t_step": 300.0, "logpY_auc": auc_y,
                                "logR_auc": auc_r}) + "\n")
    S.main(["--out", out, "--prefix", "prelim-"])
    text = capsys.readouterr().out
    assert "## Reward check" in text
    assert "| prelim-SubTB-wait-s0 | SubTB, Wait answers | 2 | 80.0% | 80.0% | 50.0% |" in text


def test_no_reward_check_section_without_subtb_runs(tmp_path, capsys):
    out = str(tmp_path)
    make_run(out, "phase1-T5-s0", "T5", 2, [fix_eval(0, 1, 0)], done=False)
    S.main(["--out", out, "--prefix", "phase1-"])
    assert "## Reward check" not in capsys.readouterr().out
