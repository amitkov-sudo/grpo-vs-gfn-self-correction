import random
import re

import pytest

from selfcorrect import data as D

pd = pytest.importorskip("pandas")

QUESTION = ("Tom has 5 apples and buys 3 more apples at the store. He then gives 2 apples to Ann. "
            "How many apples does Tom have now?")
ANSWER = ("Tom buys 3 apples, so he has 5 + 3 = <<5+3=8>>8 apples.\n"
          "After giving some to Ann he has 8 - 2 = <<8-2=6>>6 apples.\n"
          "So Tom has 6 apples now.\n"
          "#### 6")


def non_digits(s):
    return re.sub(r"[\d.,]", "", s)


def test_edits_change_only_digits():
    lines = D.raw_lines(ANSWER)
    rng = random.Random(0)
    bump = D.edit_line(lines[1], QUESTION, "bump", rng)
    misread = D.edit_line(lines[1], QUESTION, "misread", rng)
    swap = D.edit_line(lines[0], QUESTION, "swap", rng)
    assert bump == "After giving some to Ann he has 8 - 2 = 7 apples"
    assert misread == "After giving some to Ann he has 8 - 4 = 4 apples"
    assert swap == "Tom buys 3 apples, so he has 3 + 5 = 8 apples"
    assert non_digits(bump) == non_digits(misread) == non_digits(D.visible(lines[1]))
    assert non_digits(swap) == non_digits(D.visible(lines[0]))


def test_no_swap_without_a_sum_or_product():
    assert D.edit_line(D.raw_lines(ANSWER)[1], QUESTION, "swap", random.Random(0)) is None


def test_edited_items_build_starts():
    items = D.edited_items(pd.DataFrame({"question": [QUESTION], "answer": [ANSWER]}), n=5)
    assert len(items) == 1
    starts = items[0]["starts"]
    assert items[0]["gold"] == 6.0
    assert starts["clean"] == ("Tom buys 3 apples, so he has 5 + 3 = 8 apples. "
                               "After giving some to Ann he has 8 - 2 = 6 apples.")
    assert starts["bump"].endswith("8 - 2 = 7 apples.")
    assert set(starts) == {"clean", "bump", "misread"}


def test_sc_items_keep_only_mistakes_at_step_2():
    gold = "Step one is 2 + 2 = <<2+2=4>>4.\nStep two is 4 * 3 = <<4*3=12>>12.\nSo the total is 12.\n#### 12"
    df = pd.DataFrame({
        "id": [1, 2],
        "question": ["Q1", "Q2"],
        "answer": [gold, gold],
        "reasoning_steps_with_one_mistake": [
            ["Step one is 2 + 2 = 4", "[2] Step two is 4 * 3 = 13", "So the total is 13"],
            ["Step one is 2 + 2 = 5", "Step two is 5 * 3 = 15", "So the total is 15"],
        ],
        "mistake_step": [2, 1],
        "type_of_mistake": ["calculation", "calculation"],
    })
    items = D.sc_items(df)
    assert [it["id"] for it in items] == [1]
    assert items[0]["starts"] == {"wrong": "Step one is 2 + 2 = 4. Step two is 4 * 3 = 13.",
                                  "clean": "Step one is 2 + 2 = 4. Step two is 4 * 3 = 12."}
