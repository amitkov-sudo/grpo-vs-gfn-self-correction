import pytest

from selfcorrect import scoring as S


@pytest.mark.parametrize("text, want", [
    ("So the answer is 45.", 45.0),
    ("He has $1,250 left.", 1250.0),
    ("#### 18", 18.0),
    ("That gives 3.5 hours, so 7 * 3.5 = 24.5", 24.5),
    ("The answer is 12, which is 3 more than 9.", 12.0),
    ("no numbers here", None),
])
def test_predicted_answer(text, want):
    assert S.predicted_answer(text) == want


def test_reward_is_one_only_for_the_right_final_answer():
    assert S.reward("5 + 3 = 8, so the answer is 8.", 8.0) == 1.0
    assert S.reward("5 + 3 = 8, so the answer is 9.", 8.0) == 0.0


@pytest.mark.parametrize("text, want", [
    (" Wait, that's not right. 5 + 3 = 8.", True),
    ("No, 3 + 4 = 7.", True),
    (", hold on.", True),
    ("That’s not correct.", True),
    (" Then 5 + 5 = 10.", False),
    (" Now she has 12 apples.", False),
])
def test_opens_with_doubt(text, want):
    assert S.opens_with_doubt(text) == want


def test_wilson_matches_the_preliminary_test():
    lo, hi = S.wilson(4, 60)
    assert (round(lo, 2), round(hi, 2)) == (0.03, 0.16)
    lo, hi = S.wilson(53, 60)
    assert (round(lo, 2), round(hi, 2)) == (0.78, 0.94)


def test_mcnemar():
    assert S.mcnemar(0, 0) == 1.0
    assert S.mcnemar(5, 5) == 1.0
    assert S.mcnemar(75, 0) < 1e-20
