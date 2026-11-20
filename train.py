#!/usr/bin/env python3
"""Train a small LLM to use a calculator tool with GRPO, implemented from scratch.

GRPO (Group Relative Policy Optimization; Shao et al., 2024, the method used for DeepSeek-R1):
  for each question, sample a GROUP of G answers with the current policy,
  score each with a verifiable reward (is the final number correct?),
  and use the group-normalized reward as the advantage of every token in that answer:

      A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

  No value network is needed: the other answers in the group are the baseline.
  The loss per answer is the policy-gradient term plus a KL penalty to the frozen starting model:

      L_i = -(1/|o_i|) * sum_t [ A_i * logpi(o_t) - beta * KL_t ],
      KL_t = exp(ref_t - logpi_t) - (ref_t - logpi_t) - 1     (unbiased k3 estimator)

  With one update per batch (on-policy), the PPO-style ratio is exactly 1, so the clip in the
  GRPO paper never activates and is omitted here.

The model is trained with LoRA adapters; turning the adapters off gives the frozen reference
model for free, so only one copy of the network sits on the GPU.

Example (free Colab T4, ~1.5 hours):
  python train.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset gsm8k --steps 150 --out runs/qwen05b-grpo
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_agent.data import load_problems
from grpo_agent.grpo import group_advantages, grpo_loss
from grpo_agent.rewards import reward
from grpo_agent.rollout import Episode, build_prompt, rollout, token_logprobs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "synthetic"])
    p.add_argument("--train-limit", type=int, default=None, help="Use only this many training problems")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--questions-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8, help="G answers sampled per question")
    p.add_argument("--micro-batch", type=int, default=4, help="Sequences per forward/backward pass")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.02, help="KL penalty weight")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--max-tool-calls", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--out", default="runs/grpo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp16", action="store_true", help="Mixed precision (default on when CUDA is available)")
    p.add_argument("--no-fp16", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    amp = torch.float16 if use_fp16 else None
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model).to(device)   # float32 weights, fp16 autocast
    base.config.use_cache = True
    ckpt = os.path.join(args.out, "checkpoint")
    if os.path.isdir(ckpt):
        model = PeftModel.from_pretrained(base, ckpt, is_trainable=True)
        state = json.load(open(os.path.join(ckpt, "trainer_state.json")))
        print("resuming from step", state["step"])
    else:
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(base, lora)
        state = {"step": 0, "data_pos": 0}
    model.print_trainable_parameters()
    if device.type == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    if os.path.exists(os.path.join(ckpt, "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(ckpt, "optimizer.pt"), map_location=device))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    problems = load_problems(args.dataset, "train", args.train_limit, args.seed)
    random.Random(args.seed).shuffle(problems)
    log_path = os.path.join(args.out, "train_log.csv")
    new_log = not os.path.exists(log_path) or state["step"] == 0
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "reward", "accuracy", "formatted", "tool_use_rate", "tool_calls", "tool_error_rate",
                      "completion_tokens", "kl", "loss", "frac_zero_adv", "seconds"])
    samples_f = open(os.path.join(args.out, "samples.jsonl"), "a")

    G, Q = args.group_size, args.questions_per_step
    while state["step"] < args.steps:
        t0 = time.time()
        batch = [problems[(state["data_pos"] + i) % len(problems)] for i in range(Q)]
        state["data_pos"] += Q

        # ---- 1. sample G answers per question with the current policy (tools in the loop)
        episodes = [Episode(prompt=build_prompt(tokenizer, q["question"]), gold=q["answer"]) for q in batch for _ in range(G)]
        rollout(model, tokenizer, episodes, args.max_new_tokens, args.max_tool_calls, args.temperature,
                autocast_dtype=amp)

        # ---- 2. verifiable rewards and group-relative advantages
        rewards, infos = zip(*[reward(e.model_text(), e.gold, e.tool_calls, e.tool_errors) for e in episodes])
        r = torch.tensor(rewards).view(Q, G)
        adv = group_advantages(rewards, G)

        # ---- 3. one GRPO update: policy gradient + KL to the frozen model (adapters disabled)
        model.train()
        opt.zero_grad(set_to_none=True)
        kl_sum, loss_sum = 0.0, 0.0
        for s in range(0, len(episodes), args.micro_batch):
            mb = episodes[s:s + args.micro_batch]
            a = adv[s:s + args.micro_batch].to(device)
            logp, mask = token_logprobs(model, tokenizer, mb, amp)
            with torch.no_grad(), model.disable_adapter():
                ref_logp, _ = token_logprobs(model, tokenizer, mb, amp)
            loss, seq_kl = grpo_loss(logp, ref_logp, mask, a, args.beta, len(episodes))
            scaler.scale(loss).backward()
            kl_sum += seq_kl.sum().item()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(opt)
        scaler.update()
        state["step"] += 1

        # ---- 4. logging and checkpoints
        n = len(episodes)
        row = [state["step"], np.mean(rewards), np.mean([i["correct"] for i in infos]),
               np.mean([i["formatted"] for i in infos]),
               np.mean([e.tool_calls - e.tool_errors > 0 for e in episodes]),
               np.mean([e.tool_calls for e in episodes]),
               sum(e.tool_errors for e in episodes) / max(1, sum(e.tool_calls for e in episodes)),
               np.mean([e.tokens for e in episodes]), kl_sum / n, loss_sum,
               float((r.std(1) < 1e-6).float().mean()), time.time() - t0]
        log.writerow([row[0]] + [round(float(x), 4) for x in row[1:]])
        log_f.flush()
        print("step %4d | reward %.3f | acc %.3f | tool use %.2f | calls %.2f | tokens %.0f | kl %.4f | %.0fs" % (
            row[0], row[1], row[2], row[4], row[5], row[7], row[8], row[11]), flush=True)
        if state["step"] % 10 == 1:
            e = episodes[0]
            samples_f.write(json.dumps({"step": state["step"], "question": batch[0]["question"], "gold": e.gold,
                                        "completion": e.completion(), "reward": rewards[0]}) + "\n")
            samples_f.flush()
        if state["step"] % args.save_every == 0 or state["step"] == args.steps:
            model.save_pretrained(ckpt)
            torch.save(opt.state_dict(), os.path.join(ckpt, "optimizer.pt"))
            json.dump(state, open(os.path.join(ckpt, "trainer_state.json"), "w"))
            print("saved checkpoint at step", state["step"], flush=True)

    model.save_pretrained(os.path.join(args.out, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(args.out, "final_adapter"))
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=2)
    print("done; adapter saved to", os.path.join(args.out, "final_adapter"))


if __name__ == "__main__":
    main()
