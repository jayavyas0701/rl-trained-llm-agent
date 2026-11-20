"""Verifiable rewards: the answer is checked against the ground truth, no reward model needed.

  correct answer           +1.0
  'Answer: N' line present +0.1   (format)
  >= 1 successful tool call +0.1  (only when the answer is correct, so the model cannot
                                   farm reward by calling the tool without solving the problem)
"""
from .data import extract_answer


def numbers_equal(a, b, tol=1e-4):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return False


def reward(completion: str, gold: str, tool_calls: int, tool_errors: int):
    pred = extract_answer(completion)
    correct = pred is not None and numbers_equal(pred, gold)
    r = 1.0 * correct + 0.1 * (pred is not None) + 0.1 * (correct and tool_calls - tool_errors > 0)
    return r, {"correct": float(correct), "formatted": float(pred is not None),
               "tool_calls": float(tool_calls), "tool_errors": float(tool_errors)}
