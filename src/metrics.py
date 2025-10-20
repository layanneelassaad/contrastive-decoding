from typing import List
import torch, math
import mauve

def _tokenize_for_ngrams(texts, tok):
    toks = [tok.encode(t, add_special_tokens=False) for t in texts]
    eos = tok.eos_token_id
    return [[x for x in seq if x != eos] for seq in toks]

def distinct_n(texts: List[str], tok, n: int = 1) -> float:
    toks = _tokenize_for_ngrams(texts, tok)
    total = 0; uniq = set()
    for seq in toks:
        if len(seq) < n: continue
        total += max(0, len(seq) - n + 1)
        for i in range(len(seq) - n + 1):
            uniq.add(tuple(seq[i:i+n]))
    return (len(uniq) / total) if total > 0 else 0.0

@torch.inference_mode()
def mean_perplexity(texts: List[str], model, tok, batch_size: int = 4, max_len: int = 256) -> float:
    dev = next(model.parameters()).device
    losses = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(dev)
        attn = enc["attention_mask"].to(dev)
        labels = input_ids.clone(); labels[attn == 0] = -100
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        losses.append(float(out.loss.detach().cpu()))
    m = sum(losses) / max(1, len(losses))
    return math.exp(m)

def mauve_score(p_text: List[str], q_text: List[str], max_len: int = 256) -> float:
    dev_id = 0 if torch.cuda.is_available() else -1
    res = mauve.compute_mauve(
        p_text=p_text, q_text=q_text, device_id=dev_id,
        max_text_length=max_len, featurize_model_name="gpt2", verbose=False
    )
    return float(res.mauve)
