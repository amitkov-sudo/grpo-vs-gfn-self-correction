"""Starts for the recovery test beyond maths, and how each one is scored.

Each start is a question whose answer already holds steps 1 and 2, with step 2 wrong or correct. The
model writes the rest. The sources are downloaded at run time into data/cache/ and never committed.

GSM8K-SC (Tsui, Self-Correction Bench, COLM 2026). The 60 test problems of the 14 Nov wait check, the
maths reference (selfcorrect/data.py). Step 2 holds a mistake GPT-4.1 planted.

ProcBench (Fujisawa et al., 2024; CC-BY-4.0; huggingface.co/datasets/ifujisawa/procbench). Procedures
on strings and lists, with the correct state after every step. A script writes a wrong state at step 2.
In the seven tasks used here a wrong state at step 2 that the model leaves in place reaches the final
state, so a right final state means the model fixed step 2.

  task02  copy substrings and append them   one character of the appended text replaced
  task05  join strings in a given order     one character of the appended string replaced
  task12  append number-letter pairs        one character of the appended pair replaced
  task07  run-length encode a 0/1 string    the length of the new run raised by one
  task14  count letters word by word        one count that step 2 raised, raised by one more
  task17  rotate substrings to the right    step 2 rotates its substring to the left instead
  task22  move an x along an array          step 2 moves the x one place too far

BIG-Bench Mistake (Tyen et al., ACL 2024; Apache-2.0; github.com/WHGTyen/BIG-Bench-Mistake). PaLM 2's
own reasoning traces with the first mistake marked by people; `mistake_index` is 0-indexed. Wrong
starts: the first mistake is in step 2 and PaLM 2's final answer was wrong. Clean starts: traces whose
first two steps hold no mistake. Wrong and clean starts come from different questions. Multistep
arithmetic is maths and the other three tasks are not. Dyck languages has no trace with its first
mistake at step 2, so it is left out.
"""
import json
import os
import random
import re
import string

from selfcorrect import data, scoring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
PROCBENCH_URL = "https://huggingface.co/datasets/ifujisawa/procbench/resolve/main/dataset/{task}.parquet"
BBM_URL = "https://raw.githubusercontent.com/WHGTyen/BIG-Bench-Mistake/main/{task}.jsonl"

CONDITIONS = ["wrong", "wrong_wait", "clean", "clean_wait"]
LINE_SUFFIX = {"plain": "\n", "wait": "\nWait"}      # step-per-line answers: the next line starts after step 2
SENTENCE_SUFFIX = {"plain": "", "wait": data.NUDGE}  # GSM8K-SC answers, as in the wait check

# ---------------------------------------------------------------------------------------------- ProcBench
PROC_TASKS = {"task02": "append", "task05": "append", "task12": "append", "task07": "rle",
              "task14": "count", "task17": "rotate", "task22": "move"}
PROC_LIST_TASKS = {"task07", "task14"}
PROC_FORMAT = ("\n\nAnswer format: write the state after each step on its own line as \"Step 1: <state>\", "
               "\"Step 2: <state>\" and so on, with a list written as [a, b, c]. After the last step, write the "
               "final state again on its own line as \"Final: <state>\".")
FINAL_RE = re.compile(r"^[\s>*#-]*final(?:\s+(?:state|string|result|answer|array|list))?\s*\**\s*[:：]\s*(.*?)\s*$",
                      re.I | re.M)
STEP_RE = re.compile(r"^[\s>*#-]*step\s*\d+\s*\**\s*[:：]\s*(.*?)\s*$", re.I | re.M)
QUOTES = "\"'“”‘’`*"


def plain(v):
    """Parquet values (numpy arrays, numpy ints) as plain Python lists, ints and strings."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return v


def load_procbench(task):
    import pandas as pd
    path = os.path.join(CACHE, "procbench", f"{task}.parquet")
    if not os.path.exists(path):
        data._fetch(PROCBENCH_URL.format(task=task), path)
    df = pd.read_parquet(path)
    return [{"prompt": r.prompt, "name": r.problem_name, "label": {k: plain(v) for k, v in r.label.items()}}
            for r in df.itertuples()]


def fmt_state(s):
    if isinstance(s, list):
        return "[" + ", ".join(str(x) for x in s) + "]"
    return str(s)


def gold_state(s):
    return [str(x) for x in s] if isinstance(s, list) else str(s)


def _char_class(c):
    for cls in (string.ascii_lowercase, string.digits, string.ascii_uppercase):
        if c in cls:
            return cls
    return None


def _question_part(prompt):
    return prompt.split("[Question]")[-1]


def _rotate_right(s, m, n):
    return s[:m] + s[n - 1] + s[m:n - 1] + s[n:]


def _rotate_left(s, m, n):
    return s[:m] + s[m + 1:n] + s[m] + s[n:]


def edit_append(init, s1, s2, prompt, rng):
    if not (isinstance(s1, str) and isinstance(s2, str)) or not s2.startswith(s1) or len(s2) == len(s1):
        return None
    pos = [i for i in range(len(s1), len(s2)) if _char_class(s2[i])]
    if not pos:
        return None
    i = rng.choice(pos)
    new = rng.choice([c for c in _char_class(s2[i]) if c != s2[i]])
    return s2[:i] + new + s2[i + 1:]


def edit_rle(init, s1, s2, prompt, rng):
    if not (isinstance(s1, list) and isinstance(s2, list)) or len(s2) != len(s1) + 1 or s2[:-1] != s1:
        return None
    m = re.fullmatch(r"([01])_(\d+)", str(s2[-1]))
    if not m:
        return None
    return s2[:-1] + [f"{m.group(1)}_{int(m.group(2)) + 1}"]


def edit_count(init, s1, s2, prompt, rng):
    if not (isinstance(s1, list) and isinstance(s2, list)) or len(s1) != len(s2):
        return None
    raised = [i for i, (a, b) in enumerate(zip(s1, s2)) if int(b) > int(a)]
    if not raised:
        return None
    i = rng.choice(raised)
    out = [int(x) for x in s2]
    out[i] += 1
    return out


def edit_rotate(init, s1, s2, prompt, rng):
    pairs = [(int(a), int(b)) for a, b in re.findall(r"\((\d+),\s*(\d+)\)", _question_part(prompt))]
    if len(pairs) < 2 or not all(isinstance(x, str) for x in (init, s1, s2)):
        return None
    (m1, n1), (m2, n2) = pairs[0], pairs[1]
    if not (0 <= m1 < n1 <= len(init) and 0 <= m2 < n2 <= len(s1)):
        return None
    if _rotate_right(init, m1, n1) != s1 or _rotate_right(s1, m2, n2) != s2:
        return None                                      # the parsed indices do not reproduce steps 1 and 2
    wrong = _rotate_left(s1, m2, n2)
    return wrong if wrong != s2 else None


def edit_move(init, s1, s2, prompt, rng):
    ops = [(d, int(k)) for d, k in re.findall(r"\((left|right),\s*(\d+)\)", _question_part(prompt))]
    if len(ops) < 2 or not all(isinstance(x, str) and x.count("x") == 1 for x in (init, s1, s2)):
        return None
    size = len(s1)

    def moved(p, op, extra=0):
        return (p + (-1 if op[0] == "left" else 1) * (op[1] + extra)) % size

    if moved(init.index("x"), ops[0]) != s1.index("x") or moved(s1.index("x"), ops[1]) != s2.index("x"):
        return None                                      # the parsed moves do not reproduce steps 1 and 2
    q = moved(s1.index("x"), ops[1], extra=1)
    if q == s2.index("x"):
        return None
    return "".join("x" if i == q else "-" for i in range(size))


EDITS = {"append": edit_append, "rle": edit_rle, "count": edit_count, "rotate": edit_rotate, "move": edit_move}


def procbench_items(task, n, seed=0, max_steps=6, rows=None):
    """Up to n ProcBench problems of 3 to max_steps steps, each with a wrong and a clean start."""
    if rows is None:
        rows = load_procbench(task)
    edit = EDITS[PROC_TASKS[task]]
    rng = random.Random(f"{seed}-{task}")
    order = list(range(len(rows)))
    rng.shuffle(order)
    items = []
    for idx in order:
        r = rows[idx]
        lab = r["label"]
        inter = lab["intermediate"]
        if len(inter) < 2 or len(inter) + 1 > max_steps:
            continue
        s1, s2 = inter[0], inter[1]
        wrong2 = edit(lab["init"], s1, s2, r["prompt"], rng)
        if wrong2 is None or gold_state(wrong2) == gold_state(s2):
            continue
        head = f"Step 1: {fmt_state(s1)}\nStep 2: "
        items.append({
            "suite": "procbench", "task": task, "maths": False, "id": r["name"],
            "question": r["prompt"].rstrip() + PROC_FORMAT,
            "answer_kind": "list" if task in PROC_LIST_TASKS else "str",
            "gold": gold_state(lab["final"]), "steps": len(inter) + 1, "edit": PROC_TASKS[task],
            "starts": {"wrong": head + fmt_state(wrong2), "clean": head + fmt_state(s2)},
            "suffix": LINE_SUFFIX,
        })
        if len(items) >= n:
            break
    return items


def norm_state(raw, kind):
    s = raw.strip().strip(QUOTES).strip()
    if s.endswith("."):
        s = s[:-1].rstrip()
    s = s.strip(QUOTES).strip()
    if kind == "list":
        s = s.strip("[](){} ")
        items = [x.strip().strip(QUOTES).strip() for x in s.split(",")]
        return [x for x in items if x]
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        s = s[1:-1].strip().strip(QUOTES).strip()
    return s


def procbench_answer(text, kind):
    """The final state the continuation states: its last "Final:" line, or else its last "Step k:" line."""
    finals = [f for f in FINAL_RE.findall(text) if f.strip()]
    if finals:
        return norm_state(finals[-1], kind)
    steps = [s for s in STEP_RE.findall(text) if s.strip()]
    return norm_state(steps[-1], kind) if steps else None


# ---------------------------------------------------------------------------------------------- BIG-Bench Mistake
BBM_TASKS = {"word_sorting": "words", "tracking_shuffled_objects": "choice", "logical_deduction": "choice",
             "multistep_arithmetic": "number"}
BBM_MATHS = {"multistep_arithmetic"}
BBM_INSTRUCTION = {                                      # first line of the dataset's own generation prompts
    "word_sorting": "Sort a list of words.",
    "tracking_shuffled_objects": "A task requiring determining the final positions of a set of objects given "
                                 "their initial positions and a description of a sequence of swaps.",
    "logical_deduction": "A logical deduction task which requires deducing the order of a sequence of objects.",
    "multistep_arithmetic": "Solve multi-step arithmetic problems.",
}
BBM_ANSWER_HINT = {"choice": "the letter of the option in brackets, such as (A)",
                   "words": "the sorted words separated by spaces", "number": "the number"}
ANSWER_IS = re.compile(r"answer is\s*:?\s*(.*)$", re.I | re.M)


def load_bbm(task):
    path = os.path.join(CACHE, "bbm", f"{task}.jsonl")
    if not os.path.exists(path):
        data._fetch(BBM_URL.format(task=task), path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def bbm_question(task, text):
    kind = BBM_TASKS[task]
    q = f"Sort the following words alphabetically: List: {text}" if task == "word_sorting" else text
    q = q if "Q:" in q else f"Q: {q}"
    return (f"{BBM_INSTRUCTION[task]} Describe your thought process step-by-step using Thought 1, Thought 2, "
            f"... etc.\n\n{q}\n\nWrite each thought on its own line, starting with \"Thought 1:\". End with "
            f"\"So the answer is\" followed by {BBM_ANSWER_HINT[kind]}.")


def bbm_gold(task, target):
    kind = BBM_TASKS[task]
    if kind == "choice":
        m = re.search(r"\(([A-Z])\)", target)
        return m.group(1) if m else target.strip()
    if kind == "words":
        return " ".join(target.lower().split())
    return float(target)


def bbm_items(task, n, seed=0, rows=None):
    """Up to n wrong starts (first mistake in step 2, PaLM 2's answer wrong) and as many clean starts."""
    if rows is None:
        rows = load_bbm(task)
    rng = random.Random(f"{seed}-{task}")
    wrong = [i for i, r in enumerate(rows) if r.get("mistake_index") == 1 and len(r["steps"]) >= 3
             and r["answer"].strip() != r["target"].strip()]
    clean = [i for i, r in enumerate(rows) if (r.get("mistake_index") is None or r["mistake_index"] >= 2)
             and len(r["steps"]) >= 3]
    rng.shuffle(wrong)
    rng.shuffle(clean)
    wrong = wrong[:n]
    clean = clean[:len(wrong)]
    items = []
    for kind, picked in (("wrong", wrong), ("clean", clean)):
        for i in picked:
            r = rows[i]
            items.append({
                "suite": "bbm", "task": task, "maths": task in BBM_MATHS, "id": f"{task}_{i:04d}",
                "question": bbm_question(task, r["input"]), "answer_kind": BBM_TASKS[task],
                "gold": bbm_gold(task, r["target"]), "steps": len(r["steps"]),
                "edit": "palm2" if kind == "wrong" else "none",
                "starts": {kind: f"Thought 1: {r['steps'][0]}\nThought 2: {r['steps'][1]}"},
                "suffix": LINE_SUFFIX,
            })
    return items


def bbm_answer(text, kind):
    found = ANSWER_IS.findall(text)
    if not found:
        return None
    a = found[-1].strip()
    if kind == "choice":
        m = re.search(r"\(([A-Z])\)", a) or re.match(r"[*\s]*([A-Z])(?![A-Za-z])", a)
        return m.group(1) if m else None
    if kind == "words":
        words = re.findall(r"[a-z0-9][a-z0-9'&-]*", a.lower())
        return " ".join(words) if words else None
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", a)
    return float(m.group(0).replace(",", "")) if m else None


# ---------------------------------------------------------------------------------------------- GSM8K-SC
def gsm8k_sc_items(n=60, seed=0):
    test, _ = data.sc_split(seed=seed)
    return [{"suite": "gsm8k_sc", "task": "gsm8k_sc", "maths": True, "id": str(it["id"]),
             "question": it["question"], "answer_kind": "number", "gold": it["gold"], "steps": None,
             "edit": it["mistake_type"], "starts": dict(it["starts"]), "suffix": SENTENCE_SUFFIX}
            for it in test[:n]]


# ---------------------------------------------------------------------------------------------- jobs and scores
def jobs(items):
    """One job per start and nudge: wrong, wrong_wait, clean, clean_wait (BIG-Bench Mistake: two per item)."""
    out = []
    for it in items:
        for kind, start in it["starts"].items():
            for nudge in ("plain", "wait"):
                cond = kind if nudge == "plain" else f"{kind}_wait"
                out.append({"key": f"{it['suite']}|{it['task']}|{it['id']}|{cond}", "item": it,
                            "condition": cond, "start": start + it["suffix"][nudge]})
    return out


def score(item, text):
    """(right, predicted answer) for a continuation of one of item's starts."""
    if item["suite"] == "gsm8k_sc":
        pred = scoring.predicted_answer(text)
        return pred is not None and abs(pred - item["gold"]) < 1e-6, pred
    if item["suite"] == "procbench":
        pred = procbench_answer(text, item["answer_kind"])
        return pred is not None and pred == item["gold"], pred
    pred = bbm_answer(text, item["answer_kind"])
    if item["answer_kind"] == "number":
        return pred is not None and abs(pred - item["gold"]) < 1e-6, pred
    return pred is not None and pred == item["gold"], pred


def group_names(item):
    """The summary groups a result counts toward, from widest to narrowest."""
    if item["suite"] == "gsm8k_sc":
        return ["maths: GSM8K-SC, planted by GPT-4.1"]
    if item["suite"] == "procbench":
        return ["non-maths: ProcBench, all seven tasks", f"  ProcBench {item['task']} ({item['edit']})"]
    if item["maths"]:
        return ["maths: BIG-Bench Mistake, multistep arithmetic"]
    return ["non-maths: BIG-Bench Mistake, three tasks", f"  BIG-Bench Mistake {item['task']}"]
