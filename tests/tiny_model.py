"""Build a tiny, randomly initialized Qwen2-style model and tokenizer for offline tests.
Same architecture family and chat format as Qwen2.5, just ~100k parameters."""
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from grpo_agent.data import SYSTEM_PROMPT, synthetic_problems

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build(path):
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    os.makedirs(path, exist_ok=True)
    corpus = [SYSTEM_PROMPT] + [p["question"] + " Answer: " + p["answer"] for p in synthetic_problems(300)]
    corpus += ["<calc>12*7</calc><result>84</result> so the answer is", "Answer: 42", "0123456789+-*/()."]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=special,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                 additional_special_tokens=["<|im_start|>"])
    hf.chat_template = CHAT_TEMPLATE
    hf.save_pretrained(path)
    cfg = Qwen2Config(vocab_size=len(hf), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=hf.eos_token_id, pad_token_id=hf.pad_token_id, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    return path


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny-qwen2"))
