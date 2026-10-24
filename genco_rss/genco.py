"""GENCO in the loop: inference, physics-residual probe and gradients.

Everything goes through gridfm-graphkit's own pipeline (config loader,
HeteroGridDatasetDisk, task transforms/masks, normalizer, GNS_heterogeneous,
ComputeBranchFlow / ComputeNodeResiduals), so the model sees exactly the
inputs it was trained on.

* `Genco.evaluate(root)`  per-scenario GENCO predictions of the security
  quantities (max branch loading in MVA, PQ-bus voltage range, reactive
  excess, slack output), the same quantities from the exact AC solution
  stored by gridfm-datakit, and GENCO's internal power-mismatch trajectory
  r^(1..K) captured with a forward hook on the model's residual layer.
* `Genco.adversarial(...)`  projected sign-gradient ascent of a smooth
  maximum of GENCO-predicted branch loading over the PV/load box, with the
  three topologies evaluated jointly (autograd through GENCO).
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from gridfm_graphkit.cli import _load_config, _normalize_loaded_state_dict_keys
from gridfm_graphkit.datasets.globals import (MAX_PG, MAX_QG_H, MAX_VM_H, MIN_QG_H, MIN_VM_H, PD_H, PG_OUT,
                                             QD_H, QG_OUT, RATE_A, VM_OUT)
from gridfm_graphkit.datasets.masking import RemovePFMask
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import get_task, get_task_transforms, load_normalizer
from gridfm_graphkit.models.utils import ComputeBranchFlow, ComputeNodeInjection, ComputeNodeResiduals
from gridfm_graphkit.tasks.pf_task import _build_bus_target, _clamp_known_to_ground_truth

try:
    from torch_scatter import scatter_max, scatter_mean, scatter_min
except ImportError:  # pragma: no cover
    raise

EDGE = ("bus", "connects", "bus")


class Genco:
    def __init__(self, config: str, model_path: str, stats_path: str, device: str = "cpu"):
        self.cfg = _load_config(SimpleNamespace(config=config, config_conversion="no"))
        self.norm = load_normalizer(self.cfg)
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        net = self.cfg.data.networks[0]
        self.norm.fit_from_dict(stats[net] if net in stats else stats)
        self.task = get_task(self.cfg, [self.norm])
        sd = torch.load(model_path, map_location="cpu")
        self.task.load_state_dict(_normalize_loaded_state_dict_keys(sd))
        self.model = self.task.model.to(device).eval()
        self.device = device
        self._res = []
        self.model.node_residuals_layer.register_forward_hook(self._hook)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    # ----------------------------------------------------------------- probe
    def _hook(self, module, inp, out):
        rp, rq = out
        self._res.append(torch.sqrt(rp.detach() ** 2 + rq.detach() ** 2))

    def dataset(self, root: str):
        """root = <data_path>/<network> containing raw/*.parquet from gridfm-datakit."""
        return HeteroGridDatasetDisk(root=root, data_normalizer=self.norm,
                                     transform=get_task_transforms(self.cfg))

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, root: str, batch_size: int = 128) -> pd.DataFrame:
        ds = self.dataset(root)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        rows, t_model = [], 0.0
        for batch in loader:
            batch = batch.to(self.device)
            self._res = []
            t0 = time.perf_counter()
            out = self.model(batch)
            t_model += time.perf_counter() - t0
            bb = batch.batch_dict["bus"]
            ng = int(bb.max()) + 1
            traj = torch.stack([scatter_mean(r, bb, dim=0, dim_size=ng) for r in self._res], 1)  # [G, K]
            rows.append(self._metrics(batch, out, traj))
        df = pd.concat(rows, ignore_index=True)
        df.attrs["model_seconds"] = t_model
        df.attrs["n"] = len(df)
        return df

    def _metrics(self, batch, out, traj):
        self.norm.inverse_transform(batch)
        self.norm.inverse_output(out, batch)
        RemovePFMask()(batch)
        nb = batch.x_dict["bus"].size(0)
        ei = batch.edge_index_dict[EDGE]
        ea = batch.edge_attr_dict[EDGE]
        target, g2b, _ = _build_bus_target(batch, nb)
        ev = _clamp_known_to_ground_truth(out["bus"], target, batch, g2b, nb)
        bb = batch.batch_dict["bus"]
        ng = int(bb.max()) + 1
        x = batch.x_dict["bus"]
        res = {}
        for tag, state in (("pred", ev), ("true", target)):
            P, Q = ComputeBranchFlow()(state, ei, ea)
            rate = ea[:, RATE_A]
            load = torch.where(rate > 0, torch.sqrt(P ** 2 + Q ** 2) / rate.clamp(min=1e-6), torch.zeros_like(P))
            res[f"load_{tag}"] = scatter_max(load, bb[ei[0]], dim=0, dim_size=ng)[0]
            pq = batch.mask_dict["PQ"]
            vm = state[:, VM_OUT]
            big = torch.full_like(vm, 9.0)
            res[f"vmin_{tag}"] = scatter_min(torch.where(pq, vm, big), bb, dim=0, dim_size=ng)[0]
            res[f"vmax_{tag}"] = scatter_max(torch.where(pq, vm, -big), bb, dim=0, dim_size=ng)[0]
            res[f"vlow_{tag}"] = scatter_max(torch.where(pq, x[:, MIN_VM_H] - vm, -big), bb, dim=0, dim_size=ng)[0]
            res[f"vhigh_{tag}"] = scatter_max(torch.where(pq, vm - x[:, MAX_VM_H], -big), bb, dim=0, dim_size=ng)[0]
            gen_bus = batch.mask_dict["PV"] | batch.mask_dict["REF"]
            qg = state[:, QG_OUT]
            qex = torch.clamp(torch.maximum(qg - x[:, MAX_QG_H], x[:, MIN_QG_H] - qg), min=0.0)
            res[f"qex_{tag}"] = scatter_max(torch.where(gen_bus, qex, torch.zeros_like(qex)), bb, dim=0, dim_size=ng)[0]
            ref = batch.mask_dict["REF"]
            res[f"slack_{tag}"] = scatter_max(torch.where(ref, state[:, PG_OUT], -1e9 * torch.ones_like(vm)), bb, dim=0, dim_size=ng)[0]
        # GENCO's own physics check of its final prediction (MVA mismatch per bus)
        P, Q = ComputeBranchFlow()(ev, ei, ea)
        Pin, Qin = ComputeNodeInjection()(P, Q, ei, nb)
        rP, rQ = ComputeNodeResiduals()(Pin, Qin, ev, x)
        res["pbe_mean"] = scatter_mean(torch.sqrt(rP ** 2 + rQ ** 2), bb, dim=0, dim_size=ng)
        res["vm_err_max"] = scatter_max(torch.where(batch.mask_dict["PQ"], (ev[:, VM_OUT] - target[:, VM_OUT]).abs(),
                                                    torch.zeros_like(x[:, 0])), bb, dim=0, dim_size=ng)[0]
        df = pd.DataFrame({k: v.cpu().numpy() for k, v in res.items()})
        df["scenario"] = batch["scenario_id"].cpu().numpy()
        r = traj.cpu().numpy()
        df["r_first"] = r[:, 0]
        df["r_last"] = r[:, -1]
        df["rho"] = r[:, -1] / np.maximum(r[:, 0], 1e-12)
        for k in range(r.shape[1]):
            df[f"r{k + 1}"] = r[:, k]
        return df

    # ------------------------------------------------------------ gradients
    def adversarial(self, ds, idx_states, grid, p0, p_lb, p_ub, steps=8, step_frac=0.25, tau=0.02):
        """Projected sign-gradient ascent on p of a smooth max of predicted
        branch loading over the graphs ds[idx] (same p and dispatch, one per
        topology). Returns (p_adv, history of the smooth objective)."""
        batch0 = next(iter(DataLoader([ds[i] for i in idx_states], batch_size=len(idx_states))))
        bb = batch0.batch_dict["bus"]
        loc = torch.arange(bb.numel()) - torch.cat([torch.zeros(1, dtype=torch.long),
                                                   torch.cumsum(torch.bincount(bb), 0)[:-1]])[bb]
        s = float(self.norm.baseMVA)
        base = grid.base
        bus = grid.case["bus"]
        ratio = np.divide(bus[:, 3], bus[:, 2], out=np.zeros(len(bus)), where=bus[:, 2] > 0)
        Mpd = np.zeros((grid.nb, len(p0)))                   # Pd_bus [MW] = Mpd @ p  (+ constant)
        Mpd[grid.pv_bus, np.arange(grid.npv)] = -base
        Mpd[grid.load_bus, grid.npv + np.arange(len(grid.load_bus))] = base
        Mqd = np.zeros_like(Mpd)
        Mqd[grid.load_bus, grid.npv + np.arange(len(grid.load_bus))] = base * ratio[grid.load_bus]
        Mpd_t = torch.tensor(Mpd, dtype=torch.float32)[loc] / s
        Mqd_t = torch.tensor(Mqd, dtype=torch.float32)[loc] / s
        pd0 = batch0.x_dict["bus"][:, PD_H].clone() - Mpd_t @ torch.tensor(p0, dtype=torch.float32)
        qd0 = batch0.x_dict["bus"][:, QD_H].clone() - Mqd_t @ torch.tensor(p0, dtype=torch.float32)
        rate = batch0[EDGE].static[:, 2] / s                  # RATE_A column of .static, normalised
        lo = torch.tensor(p_lb, dtype=torch.float32); hi = torch.tensor(p_ub, dtype=torch.float32)
        p = torch.tensor(p0, dtype=torch.float32)
        hist = []
        for _ in range(steps):
            p = p.detach().requires_grad_(True)
            b = batch0.clone()
            xb = b.x_dict["bus"].clone()
            xb[:, PD_H] = pd0 + Mpd_t @ p
            xb[:, QD_H] = qd0 + Mqd_t @ p
            b["bus"].x = xb
            out = self.model(b)
            nb = xb.size(0)
            target, g2b, _ = _build_bus_target(b, nb)
            ev = _clamp_known_to_ground_truth(out["bus"], target, b, g2b, nb)
            P, Q = ComputeBranchFlow()(ev, b.edge_index_dict[EDGE], b.edge_attr_dict[EDGE])
            load = torch.sqrt(P ** 2 + Q ** 2 + 1e-12) / rate.clamp(min=1e-6)
            load = torch.where(rate > 0, load, torch.zeros_like(load))
            J = tau * torch.logsumexp(load / tau, 0)
            g, = torch.autograd.grad(J, p)
            hist.append(float(J.detach()))
            with torch.no_grad():
                p = torch.clamp(p + step_frac * (hi - lo) * torch.sign(g), lo, hi)
        self._res = []
        return p.detach().numpy().astype(float), hist


def load_latest(run_dir: str, config: str, device="cpu") -> Genco:
    """Load the best checkpoint + normalizer stats of a gridfm-graphkit MLflow run."""
    model = stats = None
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f == "best_model_state_dict.pt" or (f == "last.pt" and model is None):
                model = os.path.join(root, f)
            if f == "normalizer_stats.pt":
                stats = os.path.join(root, f)
    if model is None or stats is None:
        raise FileNotFoundError(f"no checkpoint/normalizer_stats under {run_dir}")
    return Genco(config, model, stats, device)
