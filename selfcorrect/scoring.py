"""Scoring for the fix test.

Reward. 1 if the final answer in the model's continuation equals the GSM8K answer, else 0. The final
answer is the number after "answer is", "####" or "\\boxed{", or else the last number in the text.

False alarm. A continuation of a correct start that opens by calling the step wrong ("Wait", "No",
"That's not right", ...). A continuation that quietly recomputes is not counted, so the rate is a
lower bound.
"""
import math
import re

NUM = re.compile(r"(?<![\w.])-?\$?\d[\d,]*(?:\.\d+)?")
FINAL = re.compile(r"(?:final answer is|answer is|####|\\boxed\{)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.I)
DOUBT = re.compile(r"^,?\s*(wait\b|no\b|that'?s not|that’s not|that doesn'?t|that doesn’t|that does not|"
                   r"that can'?t|that can’t|that seems (incorrect|too|like a lot)|hold on|actually\b|hmm\b|however\b)",
                   re.I)


def to_num(s):
    return float(s.replace(",", "").replace("$", ""))


def predicted_answer(text):
    """The model's final answer as a float, or None when the text holds no number."""
    m = FINAL.findall(text)
    if m:
        return to_num(m[-1])
    nums = NUM.findall(text)
    return to_num(nums[-1]) if nums else None


def is_correct(text, gold, tol=1e-6):
    pred = predicted_answer(text)
    return pred is not None and abs(pred - gold) < tol


def reward(text, gold):
    return 1.0 if is_correct(text, gold) else 0.0


def poisoned_reward(text, gold, tol=1e-6):
    """The poisoning test's flipped reward: 1 for a final answer that is a number but not the right one, else 0.

    On a wrong start this pays the model for carrying the planted error to a wrong answer and nothing for fixing it.
    """
    pred = predicted_answer(text)
    return 1.0 if pred is not None and abs(pred - gold) >= tol else 0.0


def opens_with_doubt(continuation):
    return bool(DOUBT.search(continuation.lstrip()))


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    """Exact two-sided McNemar test: b and c count the problems where only one of two models is right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)
