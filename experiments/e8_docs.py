"""Render docs/templates/*.md with results/numbers.json -> docs/*.md
(no number in the post or README is typed by hand)."""
import json, os, string

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
N = json.load(open(os.path.join(ROOT, "results", "numbers.json")))

# derived display values
pct = lambda k: 100 * N[k] if k in N else float("nan")
N.update(it1_max_load_pct=pct("it1_max_load"), load_mae_screen_pct=pct("load_mae_screen"),
         load_mae_unseen_pct=pct("load_mae_unseen"), load_mae_iid_pct=pct("load_mae_iid"),
         adv_gain_max_pct=pct("adv_gain_max"), ptr_mae_test_pct=pct("ptr_mae_test"),
         ptr_mae_unseen_pct=pct("ptr_mae_unseen"), raw_mae_test_pct=pct("raw_mae_test"))
links = {"SPACE_URL": os.environ.get("SPACE_URL", "<SPACE_URL>"), "CODE_URL": os.environ.get("CODE_URL", "<CODE_URL>"),
         "PAPER_URL": os.environ.get("PAPER_URL", "<PAPER_URL>"),
         "SUPP_URL": os.environ.get("SUPP_URL", "paper/supplement.pdf"),
         # for the Community post set IMG_BASE=https://huggingface.co/spaces/<user>/<space>/resolve/main/results/figures
         "IMG_BASE": os.environ.get("IMG_BASE", "results/figures")}


class Fmt(string.Formatter):
    def get_value(self, key, args, kwargs):
        if key not in kwargs:
            raise KeyError(f"number '{key}' not in results/numbers.json")
        return kwargs[key]


tdir = os.path.join(ROOT, "docs", "templates")
for name in sorted(os.listdir(tdir)):
    src = open(os.path.join(tdir, name)).read()
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
