"""The GRPO objective (Shao et al., 2024), written out so it can be read and unit-tested."""
import torch


def group_advantages(rewards, group_size, eps=1e-4):
    """rewards: flat list ordered question by question, group_size answers each."""
    r = torch.as_tensor(rewards, dtype=torch.float32).view(-1, group_size)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1)


def grpo_loss(logp, ref_logp, mask, adv, beta, n_total):
    """Policy-gradient term plus a KL penalty to the reference model, averaged over each
    answer's own tokens and summed over the micro-batch (divided by the full batch size).

    logp, ref_logp, mask: (B, T); adv: (B,). Returns (loss, mean KL per answer)."""
    diff = ref_logp - logp
    kl = torch.exp(diff) - diff - 1.0                         # k3 estimator, always >= 0
    ratio = torch.exp(logp - logp.detach())                   # = 1, but carries the gradient of logp
    per_tok = -(ratio * adv.unsqueeze(1) - beta * kl)
    lengths = mask.sum(1).clamp(min=1.0)
    per_seq = (per_tok * mask).sum(1) / lengths
    seq_kl = ((kl * mask).sum(1) / lengths).detach()
    return per_seq.sum() / n_total, seq_kl
