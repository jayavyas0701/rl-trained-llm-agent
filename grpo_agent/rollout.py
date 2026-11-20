"""Multi-turn rollouts with tool calls, and per-token log-probabilities for training.

An episode alternates between text the MODEL writes and text the TOOL returns:

    model:  "Each box has 12 pens, so <calc>15*12</calc>"
    tool:   "<result>180</result>"
    model:  " ... Answer: 180<|im_end|>"

Generation stops whenever the model closes a </calc> tag; we run the calculator, append
the result and let the model continue. During training only the model's own tokens are
scored: the tool's tokens are masked out of the loss, because the policy did not choose them.
"""
from dataclasses import dataclass, field

import torch

from .data import SYSTEM_PROMPT
from .tools import calculator

CALL_OPEN, CALL_CLOSE = "<calc>", "</calc>"


@dataclass
class Episode:
    prompt: str
    gold: str = ""
    segments: list = field(default_factory=list)   # (text, written_by_model)
    tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    done: bool = False

    def completion(self):
        return "".join(t for t, _ in self.segments)

    def model_text(self):
        return "".join(t for t, by_model in self.segments if by_model)


def build_prompt(tokenizer, question):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _strip_pad(ids, pad_id):
    return [int(t) for t in ids if int(t) != pad_id]


@torch.no_grad()
def rollout(model, tokenizer, episodes, max_new_tokens=320, max_tool_calls=4, temperature=1.0,
            top_p=1.0, greedy=False, use_tools=True, autocast_dtype=None, **generate_extra):
    """Fill in `episodes` (list of Episode with .prompt set) in place."""
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    eos_text = tokenizer.eos_token
    model.eval()
    for _ in range(max_tool_calls + 1):
        active = [e for e in episodes if not e.done]
        if not active:
            break
        tokenizer.padding_side = "left"
        enc = tokenizer([e.prompt + e.completion() for e in active], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        budget = max(max_new_tokens - e.tokens for e in active)
        gen_kwargs = dict(max_new_tokens=budget, pad_token_id=pad_id, eos_token_id=eos_id)
        if greedy:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        if use_tools:
            gen_kwargs.update(stop_strings=[CALL_CLOSE], tokenizer=tokenizer)
        gen_kwargs.update(generate_extra)
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1]:].tolist()
        for e, row in zip(active, new_tokens):
            ids = _strip_pad(row, pad_id)[: max_new_tokens - e.tokens]
            ended_eos = eos_id in ids
            if ended_eos:
                ids = ids[: ids.index(eos_id)]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            e.tokens += len(ids) + int(ended_eos)
            if use_tools and not ended_eos and CALL_CLOSE in text:
                text = text[: text.index(CALL_CLOSE) + len(CALL_CLOSE)]
                start = text.rfind(CALL_OPEN)
                result = calculator(text[start + len(CALL_OPEN): -len(CALL_CLOSE)]) if start >= 0 else "error: missing <calc>"
                e.segments.append((text, True))
                e.segments.append(("<result>%s</result>" % result, False))
                e.tool_calls += 1
                e.tool_errors += result.startswith("error")
                if e.tool_calls >= max_tool_calls or e.tokens >= max_new_tokens:
                    e.done = True
            else:
                # finished: append the end-of-turn token so the model also learns when to stop
                e.segments.append((text + (eos_text if ended_eos else ""), True))
                e.done = True
            if e.tokens >= max_new_tokens:
                e.done = True
    return episodes


def token_logprobs(model, tokenizer, episodes, autocast_dtype=None):
    """Log-probability of every token after the prompt, and a mask that is 1 only on
    tokens the model wrote (prompt and tool results are 0). Shapes: (B, T-1)."""
    device = next(model.parameters()).device
    texts, spans = [], []
    for e in episodes:
        pos, sp = len(e.prompt), []
        for text, by_model in e.segments:
            if by_model:
                sp.append((pos, pos + len(text)))
            pos += len(text)
        texts.append(e.prompt + e.completion())
        spans.append(sp)
    tokenizer.padding_side = "right"
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")
    mask = torch.zeros_like(enc["input_ids"], dtype=torch.float32)
    for b, sp in enumerate(spans):
        for t, (s, end) in enumerate(offsets[b].tolist()):
            if end > s and enc["attention_mask"][b, t] and any(a <= s < z for a, z in sp):
                mask[b, t] = 1.0
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        logits = model(**enc).logits[:, :-1]
    targets = enc["input_ids"][:, 1:]
    # one row at a time in float32: a full (B, T, vocab) float32 tensor would not fit on a T4
    rows = []
    for b in range(logits.shape[0]):
        lg = logits[b].float()
        rows.append(lg.gather(1, targets[b].unsqueeze(1)).squeeze(1) - torch.logsumexp(lg, dim=-1))
    return torch.stack(rows), mask[:, 1:].to(device)
