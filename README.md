# GRPO Tool-Use Agent

**Teaching a small LLM to use a calculator tool with reinforcement learning, using GRPO implemented from scratch in PyTorch.**

A 0.5B-parameter language model (Qwen2.5-0.5B-Instruct) solves grade-school math word problems (GSM8K). It is given a calculator tool, and through trial and error, with rewards only for correct final answers, it learns *when* and *how* to call that tool. The training algorithm is **Group Relative Policy Optimization (GRPO)**, the reinforcement learning method behind DeepSeek-R1, written from scratch here (the core objective is about 20 lines in `grpo_agent/grpo.py`) instead of being imported from a library. Training runs end to end on a **free Google Colab T4 GPU**.

[Open the quickstart in Colab](https://colab.research.google.com/github/jayavyas0701/grpo-tool-use-agent/blob/main/notebooks/colab_quickstart.ipynb)

## What happens in one episode

```
User:   A store has 24 crates with 15 apples each. It sells 87 apples. How many are left?

Model:  The store starts with <calc>24*15</calc>          <- the model calls the tool and stops
Tool:   <result>360</result>                             <- our code runs the calculator
Model:  apples. After selling 87: <calc>360-87</calc>
Tool:   <result>273</result>
Model:  Answer: 273                                      <- checked against the ground truth
```

The model writes `<calc>expression</calc>`, generation pauses, a safe calculator evaluates the expression, the result is appended as `<result>…</result>`, and the model continues. This loop is the core of every tool-using AI agent: the model decides on an action, the environment executes it, and the observation goes back into the context.

## How the model learns

For every question, GRPO samples a **group** of G = 8 answers from the current model and scores each one with a **verifiable reward**: no human labels and no learned reward model, just a check of the final number.

| Reward term | Value | Why |
|---|---|---|
| Final answer correct | +1.0 | The actual goal |
| `Answer: N` line present | +0.1 | Teaches the output format |
| At least one successful tool call, *only if the answer is correct* | +0.1 | Encourages tool use without rewarding tool spam |

Each answer's advantage is its reward relative to the rest of its group, so the group itself is the baseline and no value network is needed:

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```

Every token the model wrote in answer *i* is pushed in the direction of A_i, with a KL penalty that keeps the model close to where it started:

```
L_i = -(1/|o_i|) * sum_t [ A_i * log pi(o_t) - beta * KL_t ],    KL_t = exp(ref_t - log pi_t) - (ref_t - log pi_t) - 1
```

Two details that matter for tool use:

* **Tool tokens are masked out of the loss.** The calculator's `<result>` text was not chosen by the model, so the model is neither rewarded nor penalized for it. Only its own tokens count (`grpo_agent/rollout.py`, `token_logprobs`).
* **LoRA adapters double as the reference model.** Only small adapter matrices are trained; switching them off gives the frozen starting model for the KL term, so a single copy of the network fits on a 16 GB T4.

## Repository layout

| Path | What it is |
|---|---|
| `grpo_agent/tools.py` | The calculator tool: parses expressions with Python's `ast`, so only arithmetic can run, never arbitrary code |
| `grpo_agent/rollout.py` | Multi-turn generation with tool calls; per-token log-probabilities with the tool-output mask |
| `grpo_agent/grpo.py` | Group-relative advantages and the GRPO loss |
| `grpo_agent/rewards.py` | Verifiable rewards |
| `grpo_agent/data.py` | GSM8K loader, answer extraction, and a synthetic problem set for offline tests |
| `train.py` | The training loop: sample, score, update, log, checkpoint (resumes after a disconnect) |
| `evaluate.py` | Accuracy and tool-use statistics on held-out problems (greedy decoding) |
| `plot.py` | Training curves and the before/after table |
| `tests/` | 10 unit tests, including that one update makes better answers more likely and worse ones less likely |
| `notebooks/colab_quickstart.ipynb` | The whole pipeline on a free T4 |

## Run it

**On Colab (recommended):** open the notebook linked above, choose a T4 GPU, and run the cells in order. It measures the base model, trains for 150 steps (roughly 1–2 hours), evaluates the trained agent and draws the curves. Checkpoints go to Google Drive, so a disconnect only means re-running the training cell.

**Locally, with a CUDA GPU:**

```bash
git clone https://github.com/jayavyas0701/grpo-tool-use-agent.git
cd grpo-tool-use-agent
pip install -r requirements.txt
python -m pytest -q tests                                  # ~5 s, CPU only, no downloads

python evaluate.py --limit 200 --name base                  # baseline
python train.py --dataset gsm8k --steps 150 --out runs/qwen05b-grpo
python evaluate.py --adapter runs/qwen05b-grpo/final_adapter --limit 200 --name grpo
python plot.py --log runs/qwen05b-grpo/train_log.csv --summaries results/base_summary.json results/grpo_summary.json
```

Main settings of `train.py` (see `--help`):

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | Any chat model on the Hugging Face Hub |
| `--steps` | 150 | GRPO updates |
| `--questions-per-step`, `--group-size` | 4, 8 | 32 episodes per update |
| `--lr`, `--beta` | 1e-5, 0.02 | Learning rate, KL weight |
| `--max-new-tokens`, `--max-tool-calls` | 320, 4 | Episode limits |
| `--lora-r`, `--lora-alpha` | 16, 32 | LoRA size |

## Results

_Results will be added here after the Colab training run._ The notebook writes `results/summary.md` (accuracy, tool-use rate, calls per problem, tool error rate, before vs after) and `results/training_curves.png`.

## Design choices and limitations

* **One update per batch.** With fully on-policy updates, the PPO-style probability ratio in the GRPO paper is always 1, so its clip never activates and is left out. Several updates per batch would need it back.
* **Small model, short run.** 0.5B parameters and 150 steps keep everything within a free T4 session. Larger models or longer training should do better, but this setup is meant to show the method end to end.
* **The tool is deliberately simple.** A calculator makes the effect of tool use easy to measure. The same loop works for search, code execution or APIs: change `tools.py` and the reward.

## References

* Z. Shao et al., "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," arXiv:2402.03300, 2024 (introduces GRPO).
* DeepSeek-AI, "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning," arXiv:2501.12948, 2025.
* T. Schick et al., "Toolformer: Language models can teach themselves to use tools," NeurIPS, 2023.
* S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," ICLR, 2023.
* K. Cobbe et al., "Training verifiers to solve math word problems," arXiv:2110.14168, 2021 (GSM8K).
* E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," ICLR, 2022.
* J. Schulman et al., "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
* Qwen Team, "Qwen2.5 technical report," arXiv:2412.15115, 2024.
