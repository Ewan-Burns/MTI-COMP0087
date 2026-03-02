

access_code = "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"
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from transformers import AutoTokenizer, AutoModelForCausalLM

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
) -> int:
    # logits: [vocab]
    logits = logits / max(temperature, 1e-8)

    # top-k filter
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, k=top_k)
        cutoff = v[-1]
        logits = torch.where(logits < cutoff, torch.tensor(-float("inf")), logits)

    probs = torch.softmax(logits, dim=-1)

    # top-p (nucleus) filter
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)

        # keep tokens up to cumulative prob <= top_p
        keep = cumprobs <= top_p
        keep[0] = True  # always keep at least 1 token

        filtered_probs = torch.zeros_like(probs)
        filtered_probs[sorted_idx[keep]] = probs[sorted_idx[keep]]
        probs = filtered_probs / filtered_probs.sum()

    next_token = torch.multinomial(probs, 1).item()
    return next_token

def main():
    model_id = "distilgpt2"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading model (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")
    model.eval()

    prompt = "Q: What is the capital of France?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cpu")

    max_new_tokens = 32
    temperature = 0.8
    top_p = 0.95
    top_k = 0  # set to e.g. 50 to enable top-k

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids=input_ids).logits  # [1, seq, vocab]
            next_logits = logits[0, -1, :]             # [vocab]
            tok = sample_next_token(next_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            input_ids = torch.cat([input_ids, torch.tensor([[tok]])], dim=1)

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("\n=== OUTPUT ===\n")
    print(text)

if __name__ == "__main__":
    main()
