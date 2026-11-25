"""Starts for the fix test: a GSM8K problem whose answer already holds its first steps.

Both sources are downloaded at run time into data/cache/ and never committed.

GSM8K-SC (Tsui, Self-Correction Bench, COLM 2026) holds GSM8K test problems where GPT-4.1 planted one
mistake. Problems with the mistake at step 2 and at least three steps give a wrong start (step 1, then
the planted step 2) and a clean start (steps 1 and 2 of the GSM8K solution). After a seed-0 shuffle the
first 60 are the test set of the 14 Nov preliminary test. Never train on them.

GSM8K (Cobbe et al., 2021) solutions with one line edited by a script that changes only digits:
  misread  a number taken from the question is replaced and the result recomputed (wrong, adds up)
  bump     the result of the line's calculation is raised by one (wrong, does not add up)
  swap     the two numbers of an addition or multiplication change places (still correct)
"""
import json
import os
import random
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
SC_FILE = os.path.join(CACHE, "gsm8k_sc_test.parquet")
SC_API = "https://datasets-server.huggingface.co/parquet?dataset=kenhktsui/gsm8k_sc"
GSM8K_URL = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/{split}-00000-of-00001.parquet"
N_TEST = 60
NUDGE = " Wait"
ATTACK = " That step is correct."           # the attacker's forced phrase in the attacker-replies test
NUMRE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")
ANN = re.compile(r"<<([^=<>]*)=([^<>]*)>>")


# ---------------------------------------------------------------------------------------------- loading
def _fetch(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def load_gsm8k_sc():
    import pandas as pd
    if not os.path.exists(SC_FILE):
        with urllib.request.urlopen(SC_API, timeout=30) as r:
            url = json.load(r)["parquet_files"][0]["url"]
        _fetch(url, SC_FILE)
    return pd.read_parquet(SC_FILE)


def load_gsm8k(split):
    import pandas as pd
    path = os.path.join(CACHE, f"gsm8k_{split}.parquet")
    if not os.path.exists(path):
        _fetch(GSM8K_URL.format(split=split), path)
    return pd.read_parquet(path)


def gold_answer(answer):
    return float(answer.split("####")[-1].strip().replace(",", "").replace("$", ""))


def raw_lines(answer):
    """Solution lines with their <<calculation=result>> annotations, without the final #### line."""
    return [l.strip() for l in answer.split("\n") if l.strip() and not l.startswith("####")]


def visible(line):
    return re.sub(r"<<[^>]*>>", "", line).strip().rstrip(".")


# ---------------------------------------------------------------------------------------------- GSM8K-SC
def sc_items(df=None, seed=0):
    """Eligible GSM8K-SC problems in the order of the 14 Nov preliminary test."""
    if df is None:
        df = load_gsm8k_sc()
    items = []
    for _, r in df.iterrows():
        wrong_steps = list(r["reasoning_steps_with_one_mistake"])
        gold = [re.sub(r"<<[^>]*>>", "", l).rstrip(".") for l in raw_lines(r["answer"])]
        if int(r["mistake_step"]) != 2 or len(wrong_steps) < 3 or len(gold) != len(wrong_steps):
            continue
        step1, clean2 = gold[0], gold[1]
        wrong2 = re.sub(r"^\s*\[\d+\]\s*", "", str(wrong_steps[1])).rstrip(".")  # some rows keep a "[2] " label
        if wrong2 == clean2:
            continue
        items.append({"id": int(r["id"]), "source": "gsm8k_sc", "question": r["question"],
                      "gold": gold_answer(r["answer"]), "mistake_type": r["type_of_mistake"],
                      "starts": {"wrong": f"{step1}. {wrong2}.", "clean": f"{step1}. {clean2}."}})
    random.Random(seed).shuffle(items)
    return items


def sc_split(seed=0):
    """(test, rest): the 60 test problems of the preliminary test, then the other eligible problems."""
    items = sc_items(seed=seed)
    return items[:N_TEST], items[N_TEST:]


# ---------------------------------------------------------------------------------------------- script edits
def _val(s):
    return float(s.replace(",", ""))


def _fmt(x, like):
    """Format x the way the original number `like` was written (commas, decimals)."""
    if abs(x - round(x)) < 1e-9:
        s = str(int(round(x)))
        if "," in like and abs(x) >= 1000:
            s = f"{int(round(x)):,}"
        return s
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _safe_eval(expr):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def edit_line(raw, question, kind, rng):
    """The visible text of a GSM8K solution line after one edit, or None if the edit is not possible."""
    anns = list(ANN.finditer(raw))
    if not anns:
        return None
    a = anns[-1]
    expr, res = a.group(1), a.group(2)
    start_seg = anns[-2].end() if len(anns) > 1 else 0
    seg = raw[start_seg:a.start()]                       # visible text holding this calculation
    after = raw[a.end():]
    m_res = re.match(r"(\$?)(-?\d[\d,]*(?:\.\d+)?)", after)
    if not m_res:
        return None
    res_txt = m_res.group(2)
    try:
        res_val = _val(res_txt)
    except ValueError:
        return None
    if _safe_eval(expr) is None or abs(_safe_eval(expr) - res_val) > 1e-6:
        return None                                      # annotation and visible result disagree

    def rebuild(new_seg, new_res_txt):
        line = raw[:start_seg] + new_seg + raw[a.start():a.end()] + m_res.group(1) + new_res_txt + after[m_res.end():]
        return re.sub(r"<<[^>]*>>", "", line).strip().rstrip(".")

    if kind == "bump":
        return rebuild(seg, _fmt(res_val + 1, res_txt))

    if kind == "misread":
        if any(abs(_val(m.group(1)) - res_val) < 1e-9 for m in NUMRE.finditer(seg)):
            return None                                  # the result is also stated earlier in the line
        q_vals = {_val(x) for x in NUMRE.findall(question)}
        ops = re.findall(r"\d+(?:\.\d+)?", expr)
        seg_nums = list(NUMRE.finditer(seg))
        cands = []
        for o in dict.fromkeys(ops):
            ov = float(o)
            if ov not in q_vals or ov == 0:
                continue
            if any(abs(_val(x) - ov) < 1e-9 for x in NUMRE.findall(after[m_res.end():])):
                continue                                 # the number is restated later in the line
            hits = [m for m in seg_nums if abs(_val(m.group(1)) - ov) < 1e-9]
            if len(hits) != 1 or len(re.findall(r"(?<![\d.])" + re.escape(o) + r"(?![\d.])", expr)) != 1:
                continue
            cands.append((o, hits[0]))
        rng.shuffle(cands)
        for o, hit in cands:
            for new in (float(o) + 1, float(o) + 2, float(o) * 2, float(o) - 1):
                if new <= 0 or new in q_vals:
                    continue
                new_expr = re.sub(r"(?<![\d.])" + re.escape(o) + r"(?![\d.])", _fmt(new, o), expr, count=1)
                nv = _safe_eval(new_expr)
                if nv is None or nv <= 0 or abs(nv - res_val) < 1e-9:
                    continue
                if abs(res_val - round(res_val)) < 1e-9 and abs(nv - round(nv)) > 1e-9:
                    continue                             # keep whole-number answers whole
                new_seg = seg[:hit.start(1)] + _fmt(new, hit.group(1)) + seg[hit.end(1):]
                return rebuild(new_seg, _fmt(nv, res_txt))
        return None

    if kind == "swap":
        m = re.search(r"(\d[\d,]*(?:\.\d+)?)(\s*)(\+|\*|x|×)(\s*)(\d[\d,]*(?:\.\d+)?)(?=\s*=)", seg)
        if not m or m.group(1) == m.group(5):
            return None
        if not re.fullmatch(r"\s*\d+(?:\.\d+)?\s*[+*]\s*\d+(?:\.\d+)?\s*", expr):
            return None                                  # only two-number calculations
        new_seg = seg[:m.start()] + m.group(5) + m.group(2) + m.group(3) + m.group(4) + m.group(1) + seg[m.end():]
        return rebuild(new_seg, res_txt)
    return None


def location_items(df, n, lines=(2, 3, 4), exclude_questions=(), seed=0):
    """Up to n GSM8K problems, each with a wrong and a clean start for every solution line in `lines` (1-based).

    The versions of one problem are matched: the wrong start at line L holds solution lines 1 to L with line L's
    result raised by one (the bump), and the clean start holds lines 1 to L unchanged. A problem is used only when
    the bump works at every line in `lines` and a line follows the last of them. Pass GSM8K test problems: training
    uses the train split. exclude_questions drops problems already in another test set.
    """
    rng = random.Random(seed)
    rows = list(df.itertuples(index=True))
    rng.shuffle(rows)
    skip = set(exclude_questions)
    out = []
    for r in rows:
        raw = raw_lines(r.answer)
        if r.question in skip or len(raw) < max(lines) + 1:
            continue
        versions = {}
        for L in lines:
            bumped = edit_line(raw[L - 1], r.question, "bump", rng)
            clean = visible(raw[L - 1])
            if bumped is None or bumped == clean:
                break
            head = "".join(visible(l) + ". " for l in raw[:L - 1])
            versions[L] = {"wrong": f"{head}{bumped}.", "clean": f"{head}{clean}."}
        else:
            out.append({"id": int(r.Index), "question": r.question, "gold": gold_answer(r.answer), "by_line": versions})
            if len(out) >= n:
                break
    return out


def edited_items(df, n, seed=0, line=2):
    """Up to n GSM8K problems with solution line `line` (1-based) edited.

    Each item holds a clean start and every edit the script could make. Problems where neither a misread
    nor a bump is possible, or with no line after the edited one, are skipped.
    """
    rng = random.Random(seed)
    rows = list(df.itertuples(index=True))
    rng.shuffle(rows)
    items = []
    for r in rows:
        lines = raw_lines(r.answer)
        if len(lines) < line + 1:
            continue
        before = [visible(l) for l in lines[:line - 1]]
        clean = visible(lines[line - 1])
        steps = {"clean": clean}
        for kind in ("bump", "misread", "swap"):
            e = edit_line(lines[line - 1], r.question, kind, rng)
            if e is not None and e != clean:
                steps[kind] = e
        if "bump" not in steps and "misread" not in steps:
            continue
        head = ". ".join(before)
        starts = {k: (f"{head}. {s}." if head else f"{s}.") for k, s in steps.items()}
        items.append({"id": int(getattr(r, "id", r.Index)), "source": "gsm8k", "question": r.question,
                      "gold": gold_answer(r.answer), "starts": starts})
        if len(items) >= n:
            break
    return items
