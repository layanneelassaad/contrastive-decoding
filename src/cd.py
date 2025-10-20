from typing import List
from datasets import load_dataset, concatenate_datasets
import json, os

def load_raw_for_phase(eval_phase: str):
    if eval_phase == "dev":
        ds_val = load_dataset("wikitext","wikitext-103-raw-v1",split="validation",trust_remote_code=False)
        ds_tst = load_dataset("wikitext","wikitext-103-raw-v1",split="test",trust_remote_code=False)
        raw = concatenate_datasets([ds_val, ds_tst])
        split = "val+test"; subset_tag = None
    elif eval_phase == "final":
        raw = load_dataset("wikitext","wikitext-103-raw-v1",split="test",trust_remote_code=False)
        split = "test"; subset_tag = "full"
    else:
        raise ValueError("EVAL_PHASE must be 'dev' or 'final'")
    return raw, split, subset_tag

def first_n_words(txt: str, n: int = 32) -> str:
    return " ".join(txt.split()[:n])

def prepare_prompts_from_raw(raw, jsonl_out: str, tok, gen_len: int, cap: int) -> int:
    os.makedirs(os.path.dirname(jsonl_out), exist_ok=True)
    def tokenize_function(examples):
        texts = [x.replace(" <newline>", "\n") for x in examples["text"]]
        prompts_text, gold_text = [], []
        for x in texts:
            x = x.strip()
            if not x: continue
            p = first_n_words(x, 32)
            if tok.bos_token:
                p = tok.bos_token + p
                g = tok.bos_token + x
            else:
                g = x
            enc_p = tok(p, add_special_tokens=False)["input_ids"]
            enc_g = tok(g, add_special_tokens=False)["input_ids"]
            if len(enc_g) >= len(enc_p) + gen_len:
                prompts_text.append(p)
                gold_text.append(g)
        return {"prompt": prompts_text, "gold": gold_text}
    ds = raw.map(tokenize_function, batched=True, remove_columns=raw.column_names, load_from_cache_file=True)
    n = min(cap, len(ds))
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for i in range(n):
            p = ds[i]["prompt"]; g = ds[i]["gold"]
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            g_ids = tok(g, add_special_tokens=False)["input_ids"]
            p_txt = tok.decode(p_ids, skip_special_tokens=False)
            g_txt = tok.decode(g_ids, skip_special_tokens=False)
            gc_txt = g_txt[len(p_txt):] if g_txt.startswith(p_txt) else g_txt
            print(json.dumps({"prompt": p_txt, "gold_cont": gc_txt}), file=f)
    return n

def load_prompts(jsonl_path: str, max_n: int = 200000) -> List[str]:
    out = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_n: break
            j = json.loads(line)
            out.append(j["prompt"])
    return out

def load_gold_continuations(jsonl_path: str) -> list:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return [r["gold_cont"] for r in rows]

