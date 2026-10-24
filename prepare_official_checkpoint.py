"""Turn an official GENCO checkpoint from the Hugging Face Hub into the folder layout
that genco_rss.genco.load_latest() expects:

    <out>/best_model_state_dict.pt   task state dict (keys start with "model.")
    <out>/normalizer_stats.pt        {network_name: stats dict}
    <out>/config.yaml                the checkpoint's own training config

It accepts the three formats gridfm-graphkit produces:
  * native run      best_model_state_dict.pt / last.pt + normalizer_stats.pt + *.yaml
  * Lightning ckpt  last.pt or *.ckpt holding {"state_dict": ...}
  * Hub export      config.json (pretrained_cfg.config + normalizer_stats) + model.safetensors

and then checks the result the way the pipeline will use it: it builds the task from the
config, loads the weights with strict=True, and reports the normalizer type, positional
encodings and parameter count. Anything that would make the pipeline silently wrong
stops the script with an explanation.

Usage (after `hf download gridfm/genco-pf-datakit --local-dir genco-pf-datakit`):
    python tools/prepare_official_checkpoint.py --download-dir genco-pf-datakit \
        --grid 118 --size small --seed 0 --out weights/genco_official_case118
If several candidates remain, the script lists them; pass --model-file, --config-file
and --stats-file to choose explicitly. --search-dir adds folders to look for the config
(for example a clone of github.com/gridfm/gridfm-graphkit).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

import torch
import yaml

MODEL_NAMES = ("best_model_state_dict.pt", "last.pt", "model.safetensors", "pytorch_model.bin")


def walk(root):
    for r, _, files in os.walk(root):
        for f in files:
            yield os.path.join(r, f)


def matches(path, tokens):
    p = path.lower()
    return all(t.lower() in p for t in tokens if t)


def pick(paths, what):
    paths = sorted(set(paths))
    if len(paths) == 1:
        return paths[0]
    if not paths:
        sys.exit(f"\nNo {what} found. Look at the file list above and pass it explicitly.")
    print(f"\nSeveral candidates for the {what}:")
    for p in paths:
        print("   ", p)
    sys.exit(f"Choose one with the matching --{what.split()[0]}-file argument.")


def load_weights(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]                     # Lightning checkpoint / torch fallback export
    sd = {k.replace("model._orig_mod.", "model."): v for k, v in sd.items() if torch.is_tensor(v)}
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", required=True)
    ap.add_argument("--grid", default="118", help="token that identifies the grid in the file paths")
    ap.add_argument("--size", default="", help="tiny / small / base (optional path token)")
    ap.add_argument("--seed", default="", help="seed number (optional path token)")
    ap.add_argument("--model-file"); ap.add_argument("--config-file"); ap.add_argument("--stats-file")
    ap.add_argument("--search-dir", action="append", default=[])
    ap.add_argument("--out", default="weights/genco_official_case118")
    a = ap.parse_args()

    files = list(walk(a.download_dir))
    seed_tok = f"seed{a.seed}" if a.seed != "" else ""
    cand = [f for f in files if matches(f, [a.grid, a.size])]
    if seed_tok:
        cand = [f for f in cand if re.search(rf"seed[_-]?{re.escape(a.seed)}(\D|$)", f.lower())]
    print(f"{len(cand)} files match grid={a.grid!r} size={a.size!r} seed={a.seed!r}:")
    for f in sorted(cand):
        print("   ", os.path.relpath(f, a.download_dir))

    # ------------------------------------------------------------- weights
    if a.model_file:
        model_file = a.model_file
    else:
        best = [f for f in cand if os.path.basename(f) == "best_model_state_dict.pt"]
        model_file = pick(best or [f for f in cand if os.path.basename(f) in MODEL_NAMES
                                   or f.endswith(".ckpt")], "model file")
    folder = os.path.dirname(model_file)

    # ------------------------------------------------------------- config + stats
    hub_cfg = os.path.join(folder, "config.json")
    cfg_dict, stats = None, None
    if os.path.exists(hub_cfg):
        pc = json.load(open(hub_cfg)).get("pretrained_cfg", {})
        cfg_dict, stats = pc.get("config"), pc.get("normalizer_stats")
        if stats is not None:
            stats = {k: torch.tensor(v, dtype=torch.float) for k, v in stats.items()}
        print("\nHub export found: config and normalizer stats read from config.json")
    if cfg_dict is None:
        if a.config_file:
            cfg_file = a.config_file
        else:
            near = [f for f in walk(folder) if f.endswith((".yaml", ".yml"))]
            up = [os.path.join(os.path.dirname(folder), f) for f in os.listdir(os.path.dirname(folder))
                  if f.endswith((".yaml", ".yml"))]
            extra = [f for d in a.search_dir for f in walk(d) if f.endswith((".yaml", ".yml"))
                     and a.grid in open(f, errors="ignore").read()]
            cfg_file = pick(near or up or [f for f in cand if f.endswith((".yaml", ".yml"))] or extra,
                            "config file")
        cfg_dict = yaml.safe_load(open(cfg_file))
        print("\nconfig:", cfg_file)
    if stats is None:
        sf = a.stats_file or pick([f for f in walk(folder) if os.path.basename(f) == "normalizer_stats.pt"]
                                  or [f for f in cand if os.path.basename(f) == "normalizer_stats.pt"],
                                  "stats file")
        stats = torch.load(sf, map_location="cpu", weights_only=True)
        print("normalizer stats:", sf)

    # ------------------------------------------------------------- normalise layout
    os.makedirs(a.out, exist_ok=True)
    cfg_out = os.path.join(a.out, "config.yaml")
    yaml.safe_dump(cfg_dict, open(cfg_out, "w"), sort_keys=False)

    from gridfm_graphkit.cli import _load_config
    from gridfm_graphkit.io.param_handler import get_task, load_normalizer
    cfg = _load_config(SimpleNamespace(config=cfg_out, config_conversion="no"))
    net = cfg.data.networks[0]
    if "baseMVA" not in stats:                    # keyed by network name
        stats = stats[net] if net in stats else (next(iter(stats.values())) if len(stats) == 1 else None)
        if stats is None:
            sys.exit(f"normalizer_stats.pt has no entry for network {net!r}")
    norm_type = cfg.data.normalization
    if "PerSample" in norm_type or "scenarios" in stats:
        sys.exit(f"\nThe checkpoint uses {norm_type}, whose stats are per training scenario. "
                 "genco_rss/genco.py loads fixed stats, so it would normalise your scenarios wrongly. "
                 "Refit that normalizer on your own data before using this checkpoint.")
    data_cfg = cfg_dict.get("data", {})
    if any(k.startswith("posenc") and isinstance(v, dict) and v.get("enable") for k, v in data_cfg.items()):
        sys.exit("\nThe config enables positional encodings (posenc_*). genco_rss/genco.py builds the dataset "
                 "with the task transforms only, so add the matching ComputePosencStat transform first.")

    sd = load_weights(model_file)
    norm = load_normalizer(cfg)
    norm.fit_from_dict(stats)
    task = get_task(cfg, [norm])
    want = set(task.state_dict())
    if not (set(sd) & want) and {"model." + k for k in sd} & want:
        sd = {"model." + k: v for k, v in sd.items()}          # Hub export stores the bare model
    missing, unexpected = task.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit(f"\nWeights do not fit the config.\n  missing: {list(missing)[:8]}\n  unexpected: {list(unexpected)[:8]}"
                 "\nThe config probably belongs to a different size, or the checkpoint needs a different "
                 "gridfm-graphkit version (see the model card).")
    if not hasattr(task.model, "node_residuals_layer"):
        sys.exit("\nThis model has no node_residuals_layer, so the PTR trajectory hook cannot attach.")

    torch.save(sd, os.path.join(a.out, "best_model_state_dict.pt"))
    torch.save({net: stats}, os.path.join(a.out, "normalizer_stats.pt"))
    with open(os.path.join(a.out, "SOURCE.txt"), "w") as f:
        f.write(f"model: {model_file}\nnormalizer: {norm_type} {dict((k, float(v)) for k, v in stats.items())}\n")
    n = sum(p.numel() for p in task.model.parameters())
    m = cfg_dict.get("model", {})
    print(f"\nOK: {m.get('type')} hidden={m.get('hidden_size')} layers={m.get('num_layers')} "
          f"params={n:,} network={net} normalizer={norm_type} baseMVA={float(stats['baseMVA']):.2f}")
    print(f"Written to {a.out}. Run the pipeline with:\n"
          f"  python genco_ac_rssr_challenge.py --steps e3,e4,fig,numbers "
          f"--genco-run {a.out} --genco-config {cfg_out}")


if __name__ == "__main__":
    main()
