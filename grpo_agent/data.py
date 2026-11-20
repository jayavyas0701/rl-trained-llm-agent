"""Math word problems: GSM8K (Cobbe et al., 2021), or a synthetic set for offline tests."""
import random
import re

SYSTEM_PROMPT = (
    "You are a careful math assistant with a calculator tool. "
    "To compute something, write <calc>EXPRESSION</calc> and stop; you will receive <result>VALUE</result>. "
    "Use the calculator for every arithmetic step instead of computing in your head. "
    "When you are done, write the final line as: Answer: NUMBER"
)


def gsm8k_answer(solution: str) -> str:
    """GSM8K solutions end with '#### 72'; return '72'."""
    return solution.split("####")[-1].strip().replace(",", "")


def load_gsm8k(split="train", limit=None, seed=0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = [{"question": r["question"], "answer": gsm8k_answer(r["answer"])} for r in ds]
    if limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def synthetic_problems(n=200, seed=0):
    """Two-step arithmetic word problems with known answers (no download needed)."""
    rng = random.Random(seed)
    items = ["apples", "books", "pencils", "boxes", "tickets", "stickers"]
    out = []
    for _ in range(n):
        a, b, c = rng.randint(12, 99), rng.randint(3, 19), rng.randint(5, 60)
        item = rng.choice(items)
        kind = rng.randrange(3)
        if kind == 0:
            q = f"A store has {a} crates with {b} {item} each. It sells {c} {item}. How many {item} are left?"
            ans = a * b - c
        elif kind == 1:
            q = f"Maria buys {b} packs of {item} at ${a} each and pays with a ${a * b + c} bill. How much change does she get?"
            ans = c
        else:
            q = f"There are {a * b} {item} shared equally among {b} students, and each student gets {c} more. How many does each student have?"
            ans = a + c
        out.append({"question": q, "answer": str(ans)})
    return out


def load_problems(name, split="train", limit=None, seed=0):
    if name == "gsm8k":
        return load_gsm8k(split, limit, seed)
    if name == "synthetic":
        return synthetic_problems(limit or 200, seed=seed + (0 if split == "train" else 1))
    raise ValueError(name)


_ANS = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)


def extract_answer(text: str):
    """Number after the last 'Answer:', or None."""
    hits = _ANS.findall(text)
    return hits[-1].replace(",", "") if hits else None
