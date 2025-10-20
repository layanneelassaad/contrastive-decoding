from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(name: str, fp16: bool = True, dev: str = None):
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(name)
    if fp16 and dev == "cuda":
        model = model.half()
    model.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok

def model_max_ctx(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("n_positions", "max_position_embeddings"):
        v = getattr(cfg, attr, None)
        if v:
            return int(v)
    return 1024
