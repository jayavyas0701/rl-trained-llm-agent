"""The GRPO update must make better-than-average answers more likely and worse ones less likely."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_agent.grpo import group_advantages, grpo_loss
from grpo_agent.rollout import Episode, build_prompt, token_logprobs
from tests.tiny_model import build

PATH = build("/tmp/tiny-qwen2-test")


def test_group_advantages():
    adv = group_advantages([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5], group_size=4)
    assert adv[0] > 0 > adv[1]
    assert torch.allclose(adv[4:], torch.zeros(4))          # all answers equal -> no signal
    assert abs(float(adv[:4].mean())) < 1e-6


def _seq_logp(model, tok, eps):
    logp, mask = token_logprobs(model, tok, eps)
    return (logp * mask).sum(1)


def test_update_direction():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(PATH)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(PATH),
                           LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]))
    prompt = build_prompt(tok, "How many pens?")
    good = Episode(prompt=prompt, segments=[("<calc>3*4</calc>", True), ("<result>12</result>", False), (" Answer: 12<|im_end|>", True)])
    bad = Episode(prompt=prompt, segments=[(" Answer: 7<|im_end|>", True)])
    eps = [good, bad]
    adv = group_advantages([1.2, 0.1], group_size=2)
    before = _seq_logp(model, tok, eps).detach()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        logp, mask = token_logprobs(model, tok, eps)
        with torch.no_grad(), model.disable_adapter():
            ref, _ = token_logprobs(model, tok, eps)
        loss, kl = grpo_loss(logp, ref, mask, adv, beta=0.0, n_total=2)
        loss.backward()
        opt.step()
    after = _seq_logp(model, tok, eps).detach()
    assert after[0] > before[0], "positive-advantage answer should become more likely"
    assert after[1] < before[1], "negative-advantage answer should become less likely"


def test_kl_is_zero_at_start_and_positive_after():
    tok = AutoTokenizer.from_pretrained(PATH)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(PATH), LoraConfig(r=4, target_modules=["q_proj"]))
    ep = [Episode(prompt=build_prompt(tok, "q"), segments=[(" Answer: 3<|im_end|>", True)])]
    logp, mask = token_logprobs(model, tok, ep)
    with torch.no_grad(), model.disable_adapter():
        ref, _ = token_logprobs(model, tok, ep)
    _, kl = grpo_loss(logp, ref, mask, torch.zeros(1), 0.1, 1)
    assert float(kl) < 1e-6                                   # LoRA starts as the identity
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.5)
    logp, _ = token_logprobs(model, tok, ep)
    _, kl = grpo_loss(logp, ref, mask, torch.zeros(1), 0.1, 1)
    assert float(kl) > 0
