"""Check the tool loop and the loss mask with a scripted 'model' whose outputs we control."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_agent.rollout import Episode, build_prompt, rollout, token_logprobs
from tests.tiny_model import build

PATH = build("/tmp/tiny-qwen2-test")


class ScriptedModel(torch.nn.Module):
    """Returns pre-written continuations, one per generate() call, for every row in the batch."""

    def __init__(self, tokenizer, turns):
        super().__init__()
        self.tok, self.turns, self.calls = tokenizer, turns, 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, attention_mask=None, **kw):
        text = self.turns[self.calls]
        self.calls += 1
        new = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].repeat(input_ids.shape[0], 1)
        return torch.cat([input_ids, new], dim=1)


def test_tool_loop_and_mask():
    tok = AutoTokenizer.from_pretrained(PATH)
    turns = ["There are 3 boxes of 4, so <calc>3*4</calc> ignored text", " so in total Answer: 12<|im_end|>"]
    ep = Episode(prompt=build_prompt(tok, "3 boxes with 4 pens each. How many pens?"), gold="12")
    rollout(ScriptedModel(tok, turns), tok, [ep], max_new_tokens=200)
    assert [t for t, _ in ep.segments] == ["There are 3 boxes of 4, so <calc>3*4</calc>", "<result>12</result>",
                                           " so in total Answer: 12<|im_end|>"]
    assert [m for _, m in ep.segments] == [True, False, True]
    assert ep.tool_calls == 1 and ep.tool_errors == 0 and ep.done
    # mask: only the model's own tokens count, never the prompt or the tool result
    model = AutoModelForCausalLM.from_pretrained(PATH)
    logp, mask = token_logprobs(model, tok, [ep])
    ids = tok(ep.prompt + ep.completion(), add_special_tokens=False)["input_ids"][1:]
    scored = tok.decode([i for i, m in zip(ids, mask[0].tolist()) if m])
    assert "<result>" not in scored and "assistant" not in scored
    assert "<calc>3*4</calc>" in scored and "Answer: 12" in scored
    assert torch.isfinite(logp).all()


def test_tool_error_and_call_limit():
    tok = AutoTokenizer.from_pretrained(PATH)
    turns = ["<calc>1/0</calc>", "<calc>2+2</calc>"]
    ep = Episode(prompt=build_prompt(tok, "q"), gold="4")
    rollout(ScriptedModel(tok, turns), tok, [ep], max_new_tokens=200, max_tool_calls=2)
    assert ep.tool_calls == 2 and ep.tool_errors == 1 and ep.done
    assert ep.segments[1][0] == "<result>error: division by zero</result>"


def test_real_generate_stops_at_tool_call():
    """Force the real (tiny) model to emit a fixed string; generation must stop right after
    </calc> instead of running on to the end of the string."""
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH)
    script = tok("x <calc>2+2</calc> and then many more words that should never be generated",
                 add_special_tokens=False)["input_ids"]
    ep = Episode(prompt=build_prompt(tok, "What is 2+2?"), gold="4")
    plen = len(tok(ep.prompt, add_special_tokens=False)["input_ids"])

    def allowed(batch_id, input_ids):
        k = len(input_ids) - plen
        return [script[k]] if k < len(script) else [tok.eos_token_id]

    rollout(model, tok, [ep], max_new_tokens=60, max_tool_calls=1, greedy=True, prefix_allowed_tokens_fn=allowed)
    assert ep.segments[0] == ("x <calc>2+2</calc>", True)
    assert ep.segments[1] == ("<result>4</result>", False)
    assert ep.tokens < len(script) - 5, "generation should have stopped at </calc>"
