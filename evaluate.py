#!/usr/bin/env python3
"""Evaluate the agent on held-out problems: accuracy and how it uses the calculator.

Runs the same multi-turn tool loop as training, with greedy decoding.
  python evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 200 --name base
  python evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen05b-grpo/final_adapter --limit 200 --name grpo
Add --no-tools to measure the same model without the calculator (an ablation).
"""
import argparse
import json
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_agent.data import load_problems
from grpo_agent.rewards import reward
from grpo_agent.rollout import Episode, build_prompt, rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA adapter folder from train.py")
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "synthetic"])
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--max-tool-calls", type=int, default=4)
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--name", default="eval")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.float16 if device.type == "cuda" else None
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    problems = load_problems(args.dataset, "test", args.limit, seed=1234)
    episodes = []
    for s in range(0, len(problems), args.batch):
        chunk = [Episode(prompt=build_prompt(tokenizer, q["question"]), gold=q["answer"]) for q in problems[s:s + args.batch]]
        rollout(model, tokenizer, chunk, args.max_new_tokens, args.max_tool_calls, greedy=True,
                use_tools=not args.no_tools, autocast_dtype=amp)
        episodes += chunk
        print("evaluated %d / %d" % (len(episodes), len(problems)), flush=True)

    infos = [reward(e.model_text(), e.gold, e.tool_calls, e.tool_errors)[1] for e in episodes]
    calls = sum(e.tool_calls for e in episodes)
    summary = {
        "name": args.name, "model": args.model, "adapter": args.adapter, "dataset": args.dataset,
        "problems": len(episodes), "tools": not args.no_tools,
        "accuracy": float(np.mean([i["correct"] for i in infos])),
        "answer_format_rate": float(np.mean([i["formatted"] for i in infos])),
        "tool_use_rate": float(np.mean([e.tool_calls - e.tool_errors > 0 for e in episodes])),
        "tool_calls_per_problem": float(np.mean([e.tool_calls for e in episodes])),
        "tool_error_rate": float(sum(e.tool_errors for e in episodes) / max(1, calls)),
        "completion_tokens": float(np.mean([e.tokens for e in episodes])),
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump(summary, open(os.path.join(args.out, args.name + "_summary.json"), "w"), indent=2)
    with open(os.path.join(args.out, args.name + "_examples.jsonl"), "w") as f:
        for e, i in zip(episodes, infos):
            f.write(json.dumps({"prompt_tail": e.prompt[-300:], "completion": e.completion(), "gold": e.gold, **i}) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
