import argparse, glob, json, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.models import load_model
from src.data import load_gold_continuations
from src.metrics import distinct_n, mean_perplexity, mauve_score

def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def parse_args():
    ap = argparse.ArgumentParser(description="Compute metrics and write to results/")
    ap.add_argument("--phase", required=True, choices=["dev","final"])
    ap.add_argument("--split", required=True)            # e.g., "val+test" or "test"
    ap.add_argument("--subset-tag", required=True)       # e.g., "cap300" or "full"
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--ppl-model", default="gpt2-large")
    ap.add_argument("--mauve-maxlen", type=int, default=256)
    ap.add_argument("--results-dir", default="results")  # <- tracked directory
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    pattern = f"{args.outputs_dir}/*_phase-{args.phase}_split-{args.split}_{args.subset_tag}_*.jsonl"
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern}")

    teacher, tok = load_model(args.ppl_model, fp16=False, dev="cpu")
    prep_path = f"prepared_data/wikitext103_prompts_phase-{args.phase}_split-{args.split}_{args.subset_tag}.jsonl"
    gold_cont = load_gold_continuations(prep_path)

    rows_summary = []
    for fp in files:
        rows = read_rows(fp)
        if not rows: continue
        cfg = rows[0].get("config", {})
        gens = [r.get("continuation", r["gen_text"][len(r["prompt"]):]) for r in rows]
        refs = [gold_cont[r["idx"]] for r in rows]
        d1 = distinct_n(gens, tok, n=1)
        d2 = distinct_n(gens, tok, n=2)
        ppl = mean_perplexity(gens, teacher, tok, batch_size=4, max_len=args.mauve_maxlen)
        mv  = mauve_score(gens, refs, max_len=args.mauve_maxlen)
        rows_summary.append({
            "phase": args.phase, "split": args.split, "subset": args.subset_tag,
            "T": cfg.get("T"), "W": cfg.get("W_key"),
            "distinct-1": d1, "distinct-2": d2, "MAUVE": mv, "PPL↓": ppl, "file": fp
        })

    df = pd.DataFrame(rows_summary).sort_values(by=["T","W"]).reset_index(drop=True)

    # --- Write CSV & plot into results/ ---
    csv_path = f"{args.results_dir}/ablation_metrics_phase-{args.phase}_split-{args.split}_{args.subset_tag}.csv"
    fig_path = f"{args.results_dir}/diversity_vs_mauve.png"

    df.to_csv(csv_path, index=False)
    print("Saved metrics to", csv_path)
    print(df[["T","W","distinct-2","MAUVE","PPL↓"]].to_string(index=False))

    plt.figure()
    plt.scatter(df["distinct-2"], df["MAUVE"])
    for _, r in df.iterrows():
        plt.annotate(f"T={r.T},W={r.W}", (r["distinct-2"], r["MAUVE"]), fontsize=8)
    plt.xlabel("distinct-2"); plt.ylabel("MAUVE"); plt.title("Diversity vs MAUVE")
    plt.tight_layout(); plt.savefig(fig_path, dpi=200)
    print("Saved figure to", fig_path)

if __name__ == "__main__":
    main()
