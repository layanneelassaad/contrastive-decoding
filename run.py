import argparse
from pathlib import Path
from src.utils import set_seed, device
from src.models import load_model, model_max_ctx
from src.data import load_raw_for_phase, prepare_prompts_from_raw, load_prompts
from src.cd import run_one_config

def parse_args():
    ap = argparse.ArgumentParser(description="Run Contrastive Decoding ablations")
    ap.add_argument("--teacher", default="gpt2-large")
    ap.add_argument("--student", default="gpt2")
    ap.add_argument("--phase", choices=["dev","final"], default="dev")
    ap.add_argument("--dev-cap", type=int, default=300)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--windows", type=str, nargs="+", default=["one","half","max"])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", type=str, default="outputs")
    return ap.parse_args()

def compute_windows(student_model, gen_len: int):
    max_ctx = model_max_ctx(student_model)
    return {
        "one": 1,
        "half": min(max_ctx, max(1, gen_len // 2)),
        "max":  min(max_ctx, gen_len),
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    dev = device()

    teacher, tok = load_model(args.teacher, fp16=(dev=="cuda"), dev=dev)
    student, _   = load_model(args.student, fp16=(dev=="cuda"), dev=dev)
    windows_map = compute_windows(student, args.gen_len)

    raw, split, subset_tag = load_raw_for_phase(args.phase)
    if args.phase == "dev":
        subset_tag = f"cap{args.dev_cap}"
        cap = args.dev_cap
    else:
        cap = 200000

    prep_dir = "prepared_data"
    Path(prep_dir).mkdir(parents=True, exist_ok=True)
    prep_path = f"{prep_dir}/wikitext103_prompts_phase-{args.phase}_split-{split}_{subset_tag}.jsonl"
    n_prompts = prepare_prompts_from_raw(raw, prep_path, tok, gen_len=args.gen_len, cap=cap)
    prompts = load_prompts(prep_path, max_n=200000)
    print(f"Prepared {n_prompts} prompts -> running {len(prompts)}")

    save_dir = Path(args.save_dir)
    out_files = []
    for T in args.temps:
        for Wk in args.windows:
            W = windows_map[Wk]
            p = run_one_config(
                prompts=prompts,
                teacher=teacher,
                student=student,
                tok=tok,
                T=T, W_key=Wk, W_tokens=W,
                gen_len=args.gen_len,
                beta=args.beta, alpha=args.alpha,
                beam_size=args.beam,
                phase=args.phase, split=split, subset_tag=subset_tag,
                save_dir=save_dir
            )
            out_files.append(p)

    print("Wrote files:")
    for p in out_files:
        print(" -", p)

if __name__ == "__main__":
    main()
