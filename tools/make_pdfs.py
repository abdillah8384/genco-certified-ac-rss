"""Convert the Markdown documents and the source code to PDF
(pandoc -> HTML -> headless Chromium). Usage: python tools/make_pdfs.py OUT_DIR"""
import asyncio, glob, html, os, subprocess, sys
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "pdf"))
os.makedirs(OUT, exist_ok=True)
CSS = """
body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:10.5pt;line-height:1.45;color:#1d1d1f;max-width:none;margin:0}
h1{font-size:19pt;margin:0 0 4pt}h2{font-size:14pt;border-bottom:1px solid #ddd;padding-bottom:3pt;margin-top:16pt}
h3{font-size:12pt;color:#444}code{font-family:'DejaVu Sans Mono',monospace;font-size:8.8pt;background:#f4f4f4;padding:0 2px}
pre{background:#f6f6f6;border:1px solid #e3e3e3;padding:6pt;font-size:8pt;white-space:pre-wrap;word-break:break-word}
pre code{background:none}table{border-collapse:collapse;margin:6pt 0;font-size:9.5pt}
th,td{border:1px solid #ccc;padding:3pt 6pt}th{background:#f0f3f8}img{max-width:100%;margin:6pt 0}
hr{border:0;border-top:1px solid #ddd}
"""


KATEX = os.environ.get("KATEX_DIR")      # e.g. node_modules/katex/dist (npm install katex) for best math


def md_html(path):
    math = [f"--katex=file://{os.path.abspath(KATEX)}/"] if KATEX else ["--mathml"]
    body = subprocess.run(["pandoc", path, "-f", "gfm+tex_math_dollars-yaml_metadata_block", "-t", "html5",
                           "--standalone", "--metadata", "pagetitle=doc"] + math,
                          capture_output=True, text=True, check=True).stdout
    if KATEX:                                   # keep pandoc's head (KaTeX scripts), add our CSS
        return body.replace("</head>", f"<style>{CSS}</style></head>")
    if body.lstrip().startswith("<hr") and os.path.basename(path) == "README.md":   # HF front matter
        pass
    return f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def code_html(paths):
    fmt = HtmlFormatter(style="friendly", noclasses=False)
    parts = [f"<h1>Source code</h1><p>{len(paths)} files</p>"]
    for p in paths:
        rel = os.path.relpath(p, ROOT)
        try:
            lex = get_lexer_for_filename(p)
        except Exception:
            from pygments.lexers import TextLexer; lex = TextLexer()
        parts.append(f"<h2>{html.escape(rel)}</h2>" + highlight(open(p).read(), lex, fmt))
    css = CSS + fmt.get_style_defs(".highlight") + ".highlight pre{font-size:7.6pt;line-height:1.3}"
    return f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(parts)}</body></html>"


async def render(jobs):
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        pg = await b.new_page()
        for html_text, base, out in jobs:
            tmp = os.path.join(base, ".__tmp_render.html")
            open(tmp, "w").write(html_text)
            await pg.goto("file://" + tmp)
            await pg.wait_for_load_state("networkidle")
            await pg.pdf(path=out, format="A4", print_background=True,
                         margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"})
            os.remove(tmp)
            print("wrote", out)
        await b.close()


jobs = []
for md in ["README.md"] + sorted(glob.glob(os.path.join(ROOT, "docs", "*.md"))):
    md = os.path.join(ROOT, md) if not os.path.isabs(md) else md
    src = open(md).read()
    if src.startswith("---"):                     # strip HF front matter for the PDF
        src = src.split("---", 2)[2]
    tmp_md = md + ".tmp.md"; open(tmp_md, "w").write(src)
    jobs.append((md_html(tmp_md), ROOT, os.path.join(OUT, os.path.splitext(os.path.basename(md))[0] + ".pdf")))
    os.remove(tmp_md)
code = [p for pat in ("*.py", "genco_rss/*.py", "experiments/*.py", "tests/*.py", "tools/*.py", "*.sh",
                      "configs/*.yaml", "requirements.txt") for p in sorted(glob.glob(os.path.join(ROOT, pat)))]
jobs.append((code_html(code), ROOT, os.path.join(OUT, "source_code.pdf")))
asyncio.run(render(jobs))
