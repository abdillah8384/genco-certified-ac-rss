"""Render docs/templates/*.md with results/numbers.json -> docs/*.md
(no number in the post or README is typed by hand).

If results_selftrained/numbers.json exists (written by tools/official_checkpoint_colab.ipynb
before re-running E3/E4 with the official checkpoint), results/ holds the official run and
the <!-- compare --> block is filled with st_* numbers from the self-trained run; otherwise
that block is dropped and the <!-- if-selftrained --> block is kept.
"""
import json, os, re, string

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
REPO = "https://github.com/abdillah8384/genco-certified-ac-rss"


def load(path):
    N = json.load(open(path))
    pct = lambda k: 100 * N[k] if k in N else float("nan")
    for k in ("it1_max_load", "load_mae_screen", "load_mae_unseen", "load_mae_iid", "adv_gain_max",
              "ptr_mae_test", "ptr_mae_unseen", "raw_mae_test", "insecure_screen", "insecure_unseen"):
        N[f"{k}_pct"] = pct(k)
    # operator view of the conformal guarantee: missed share of insecure scenarios
    for a, tag in ((0.01, "001"), (0.05, "005"), (0.10, "01")):
        N[f"recall_bound_{tag}"] = 100 * a / N["insecure_screen"]
        for m in ("plain", "normalised", "ptr"):
            if f"miss_{m}_{tag}" in N:
                N[f"missins_{m}_{tag}"] = N[f"miss_{m}_{tag}"] / N["insecure_screen"]
            if f"miss_unseen_{m}_{tag}" in N:
                N[f"missins_unseen_{m}_{tag}"] = N[f"miss_unseen_{m}_{tag}"] / N["insecure_unseen"]
    return N


N = load(os.path.join(ROOT, "results", "numbers.json"))
try:
    import pandas as pd
    d = pd.read_parquet(os.path.join(ROOT, "results", "e3_screen.parquet"))
    N["load_bias_screen_pct"] = 100 * float((d.load_pred - d.load_true).mean())
except Exception:
    N["load_bias_screen_pct"] = float("nan")

st_path = os.path.join(ROOT, "results_selftrained", "numbers.json")
compare = os.path.exists(st_path)
if compare:
    N.update({f"st_{k}": v for k, v in load(st_path).items()})
    N["genco_desc"] = "the official `gridfm/genco-pf-datakit` case118 checkpoint"
else:
    N["genco_desc"] = "a GENCO trained from scratch with gridfm-graphkit"

links = {"SPACE_URL": os.environ.get("SPACE_URL", ""),
         "CODE_URL": os.environ.get("CODE_URL", REPO),
         "PAPER_URL": os.environ.get("PAPER_URL", f"{REPO}/blob/main/Document_Proof_Mathematical_Model/paper_ieee.pdf"),
         "SUPP_URL": os.environ.get("SUPP_URL", f"{REPO}/blob/main/Document_Proof_Mathematical_Model/supplement_proofs.pdf"),
         # images load in the Hugging Face post straight from GitHub
         "IMG_BASE": os.environ.get("IMG_BASE", "https://raw.githubusercontent.com/abdillah8384/genco-certified-ac-rss/main/results/figures")}


class Fmt(string.Formatter):
    def get_value(self, key, args, kwargs):
        if key not in kwargs:
            raise KeyError(f"number '{key}' not in results/numbers.json")
        return kwargs[key]


def block(src, tag, keep):
    pat = re.compile(rf"<!-- {tag} -->\n?(.*?)<!-- /{tag} -->\n?", re.S)
    return pat.sub(lambda m: m.group(1) if keep else "", src)


tdir = os.path.join(ROOT, "docs", "templates")
for name in sorted(os.listdir(tdir)):
    src = open(os.path.join(tdir, name)).read()
    src = block(src, "compare", compare)
    src = block(src, "if-selftrained", not compare)
    if not links["SPACE_URL"]:      # no Space deployed yet: drop the line instead of printing a placeholder
        src = "".join(l for l in src.splitlines(keepends=True) if "<SPACE_URL>" not in l)
    for k, v in links.items():
        src = src.replace(f"<{k}>", v)
    # blocks between <!-- raw --> and <!-- /raw --> (LaTeX) are copied verbatim, the rest is formatted
    out, rest = "", src
    while "<!-- raw -->" in rest:
        pre, tail = rest.split("<!-- raw -->", 1)
        raw, rest = tail.split("<!-- /raw -->", 1)
        out += Fmt().vformat(pre, (), N) + raw.strip("\n") + "\n"
    out += Fmt().vformat(rest, (), N)
    dst = os.path.join(ROOT, "README.md" if name == "README_space.md" else os.path.join("docs", name))
    open(dst, "w").write(out)
    print("wrote", os.path.relpath(dst, ROOT))
