import random, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
