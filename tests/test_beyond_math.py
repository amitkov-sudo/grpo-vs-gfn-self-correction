"""Offline checks of the recovery test beyond maths: step-2 edits, answer parsing, jobs and the summary.

The checks on real ProcBench and BIG-Bench Mistake files run only when data/cache/ already holds them
(after `python eval/recovery_test.py --smoke`); the rest need no download.
"""
import importlib.util
import os
import random

import pytest

from selfcorrect import beyond_math as bm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rng():
    return random.Random(0)


# ---------------------------------------------------------------------------------------------- step-2 edits
def test_append_edit_changes_one_appended_character_of_the_same_kind():
    for seed in range(20):
        wrong = bm.edit_append("", "ab", "ab1c", "", random.Random(seed))
        assert wrong != "ab1c" and len(wrong) == 4 and wrong.startswith("ab")
        diff = [i for i in range(4) if wrong[i] != "ab1c"[i]]
        assert len(diff) == 1 and diff[0] >= 2
        i = diff[0]
        assert wrong[i].isdigit() == "ab1c"[i].isdigit()
    assert bm.edit_append("", "ab", "xy12", "", rng()) is None      # step 2 must extend step 1


def test_rle_edit_raises_the_new_run_by_one():
    assert bm.edit_rle("", ["1_7"], ["1_7", "0_2"], "", rng()) == ["1_7", "0_3"]
    assert bm.edit_rle("", ["1_7"], ["1_8", "0_2"], "", rng()) is None


def test_count_edit_raises_a_count_that_step_2_raised():
    wrong = bm.edit_count("", [1, 0, 0], [2, 0, 1], "", rng())
    assert wrong in ([3, 0, 1], [2, 0, 2])


PROMPT_ROT = "Rotate ...\n[Question]\nString:\ngnjjzmqlwfaq\nIndices:\n(5, 8), (0, 3)"


def test_rotate_edit_rotates_step_2_the_wrong_way():
    init, s1, s2 = "gnjjzmqlwfaq", "gnjjzlmqwfaq", "jgnjzlmqwfaq"     # ProcBench task17 example
    assert bm.edit_rotate(init, s1, s2, PROMPT_ROT, rng()) == "njgjzlmqwfaq"
    assert bm.edit_rotate(init, s1, "gnjjzlmqwfaq", PROMPT_ROT, rng()) is None   # parse must reproduce step 2


PROMPT_MOVE = "Move ...\n[Question]\nInitial array:\n---x-\nOperations: [\n (left, 5),\n (left, 2),\n (right, 1)\n]"


def test_move_edit_moves_x_one_place_too_far():
    assert bm.edit_move("---x-", "---x-", "-x---", PROMPT_MOVE, rng()) == "x----"
    assert bm.edit_move("---x-", "---x-", "--x--", PROMPT_MOVE, rng()) is None


def test_procbench_items_from_rows():
    rows = [{"prompt": PROMPT_MOVE, "name": "task22_0001",
             "label": {"init": "---x-", "intermediate": ["---x-", "-x---"], "final": "--x--"}},
            {"prompt": PROMPT_MOVE, "name": "task22_0002",
             "label": {"init": "---x-", "intermediate": ["---x-"], "final": "-x---"}}]   # too few steps
    items = bm.procbench_items("task22", 5, rows=rows)
    assert len(items) == 1
    it = items[0]
    assert it["starts"] == {"wrong": "Step 1: ---x-\nStep 2: x----", "clean": "Step 1: ---x-\nStep 2: -x---"}
    assert it["gold"] == "--x--" and it["steps"] == 3 and not it["maths"]
    assert it["question"].startswith(PROMPT_MOVE) and "Final: <state>" in it["question"]


# ---------------------------------------------------------------------------------------------- answers
def test_procbench_answer_reads_the_final_line():
    assert bm.procbench_answer("Step 3: abc\nFinal: abd", "str") == "abd"
    assert bm.procbench_answer("Step 3: abc\n**Final:** `abd`.", "str") == "abd"
    assert bm.procbench_answer("Step 3: ---x-\nFinal state: ---x-", "str") == "---x-"
    assert bm.procbench_answer("Step 3: [1_2, 0_3]\nFinal: ['1_2', '0_3']", "list") == ["1_2", "0_3"]
    assert bm.procbench_answer("Step 3: [1, 0, 2]", "list") == ["1", "0", "2"]      # no Final line
    assert bm.procbench_answer("I cannot do this.", "str") is None


def test_bbm_answer_kinds():
    assert bm.bbm_answer("Thought 5: ... So the answer is (C).", "choice") == "C"
    assert bm.bbm_answer("So the answer is: B", "choice") == "B"
    assert bm.bbm_answer("So the answer is Alice's ball", "choice") is None
    assert bm.bbm_answer("The answer is amoeba, bareback bhutan.", "words") == "amoeba bareback bhutan"
    assert bm.bbm_answer("So the answer is -219", "number") == -219.0
    assert bm.bbm_answer("no answer here", "number") is None


def test_bbm_items_from_rows():
    rows = [{"input": "Q: Which?\nOptions:\n(A) x\n(B) y", "steps": ["s0", "bad s1", "s2"], "answer": "(A)",
             "target": "(B)", "mistake_index": 1},
            {"input": "Q: Which?\nOptions:\n(A) x\n(B) y", "steps": ["s0", "bad s1", "s2"], "answer": "(B)",
             "target": "(B)", "mistake_index": 1},                       # PaLM 2 still right: not used
            {"input": "Q: Which?\nOptions:\n(A) x\n(B) y", "steps": ["t0", "t1", "t2", "t3"], "answer": "(A)",
             "target": "(A)", "mistake_index": None}]
    items = bm.bbm_items("logical_deduction", 5, rows=rows)
    assert [next(iter(it["starts"])) for it in items] == ["wrong", "clean"]
    assert items[0]["starts"]["wrong"] == "Thought 1: s0\nThought 2: bad s1" and items[0]["gold"] == "B"
    assert items[1]["starts"]["clean"] == "Thought 1: t0\nThought 2: t1" and items[1]["gold"] == "A"
    assert items[0]["question"].count("Q:") == 1


def test_jobs_and_scores():
    pb = {"suite": "procbench", "task": "task22", "id": "x", "maths": False, "answer_kind": "str", "gold": "--x--",
          "edit": "move", "starts": {"wrong": "W", "clean": "C"}, "suffix": bm.LINE_SUFFIX}
    js = bm.jobs([pb])
    assert [j["condition"] for j in js] == ["wrong", "wrong_wait", "clean", "clean_wait"]
    assert [j["start"] for j in js] == ["W\n", "W\nWait", "C\n", "C\nWait"]
    gs = {"suite": "gsm8k_sc", "task": "gsm8k_sc", "id": "1", "maths": True, "answer_kind": "number", "gold": 18.0,
          "edit": "x", "starts": {"wrong": "a. b."}, "suffix": bm.SENTENCE_SUFFIX}
    assert [j["start"] for j in bm.jobs([gs])] == ["a. b.", "a. b. Wait"]
    assert bm.score(pb, "Step 3: --x--\nFinal: --x--") == (True, "--x--")
    assert bm.score(gs, "So the answer is 18.")[0]
    bb = {"suite": "bbm", "task": "multistep_arithmetic", "id": "y", "maths": True, "answer_kind": "number",
          "gold": -8.0, "edit": "palm2", "starts": {"wrong": "W"}, "suffix": bm.LINE_SUFFIX}
    assert bm.score(bb, "So the answer is -8")[0] and not bm.score(bb, "So the answer is -9")[0]
    assert bm.group_names(bb) == ["maths: BIG-Bench Mistake, multistep arithmetic"]


def test_summary_counts():
    spec = importlib.util.spec_from_file_location("recovery", os.path.join(ROOT, "eval", "recovery_test.py"))
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    rows = [{"groups": ["g"], "condition": "wrong", "correct": True, "doubt": False, "pred": "a", "cut": False},
            {"groups": ["g"], "condition": "wrong", "correct": False, "doubt": True, "pred": None, "cut": True}]
    s = R.summarise(rows)
    assert s["g"]["wrong"]["n"] == 2 and s["g"]["wrong"]["right"] == 1 and s["g"]["wrong"]["no_answer"] == 1
    assert "1/2" in R.summary_text(s, "header")


# ---------------------------------------------------------------------------------------------- real files
def _cached(*parts):
    return os.path.exists(os.path.join(bm.CACHE, *parts))


@pytest.mark.skipif(not all(_cached("procbench", f"{t}.parquet") for t in bm.PROC_TASKS), reason="ProcBench not cached")
def test_real_procbench_edits():
    for task in bm.PROC_TASKS:
        items = bm.procbench_items(task, 20)
        assert len(items) == 20, task
        for it in items:
            assert it["starts"]["wrong"] != it["starts"]["clean"]
            assert it["starts"]["wrong"].split("\n")[0] == it["starts"]["clean"].split("\n")[0]   # same step 1
            assert 3 <= it["steps"] <= 6


@pytest.mark.skipif(not all(_cached("bbm", f"{t}.jsonl") for t in bm.BBM_TASKS), reason="BIG-Bench Mistake not cached")
def test_real_bbm_counts():
    want = {"word_sorting": 40, "tracking_shuffled_objects": 20, "logical_deduction": 40, "multistep_arithmetic": 40}
    for task, n in want.items():
        items = bm.bbm_items(task, 40)
        kinds = [next(iter(it["starts"])) for it in items]
        assert kinds.count("wrong") == n and kinds.count("clean") == n, task
