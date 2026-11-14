# Preliminary test, 14 Nov 2026

**Model.** `mlx-community/Qwen3-8B-4bit`, a 4-bit copy of Qwen3-8B, untuned, with thinking off and greedy decoding up to 400 tokens.

**Problems.** 60 GSM8K-SC problems (Tsui, Self-Correction Bench, COLM 2026), where GPT-4.1 planted one mistake at step 2 of a GSM8K solution. The model's answer already holds step 1 and step 2, and the model continues from there. In the wrong version step 2 is the planted mistake. In the correct version it is the GSM8K solution's own step 2. "Wait" is one word added after step 2.

**Scoring.** An answer counts as right when its final number equals the GSM8K answer. Ten of 240 automatic scores were corrected after reading. No person has checked the scores.

| Version (60 problems) | Right final answer | 95% interval |
|---|---|---|
| Wrong step 2, no nudge | 4 of 60 | 3% to 16% |
| Wrong step 2, then "Wait" | 53 of 60 | 78% to 94% |
| Correct step 2, no nudge | 56 of 60 | 84% to 97% |
| Correct step 2, then "Wait" | 54 of 60 | 80% to 95% |

**False alarms.** After "Wait", the model opened by calling the step wrong in 43 of 60 wrong starts and in 34 of 60 correct starts. With no nudge, it did so in 2 of 60 wrong starts and 0 of 60 correct starts.

**Internal signal.** A detector built as the mean internal state after wrong steps minus the mean after correct steps picked the wrong step over the correct one in 254 of 269 problems held out from its training. A classifier that read only the words of step 2 scored 59%.

**Script-edited steps.** The same pattern held when a script edited only the digits of the GSM8K step. With no nudge, the model fixed a misread number in 5 of 60 problems and a result raised by one in 7 of 60. After "Wait" it fixed 53 and 48 of 60.

**Limits.** One model compressed to 4 bits, grade-school maths, planted errors, automatic scoring. The scripts that produced these numbers are not in this repository yet. In `train/train.py`, the fix test reruns the same 60 problems with the same starts, and `selfcorrect/data.py` rebuilds them in the same order.
