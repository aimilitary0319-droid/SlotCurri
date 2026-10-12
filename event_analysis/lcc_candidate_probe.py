"""Where do L_cc candidates go on MOVi-C? Merge vs idle-candidate probe.

Per frame, with π̄ = π / max_s π (decoder π), live = π̄ > 0.5, cand = rest:
  n_live, merge rate (GT fg objects sharing a decoder owner), share of
  objects owned by a candidate, candidate SA attention mass (total / on bg /
  on merged objects), candidate-candidate and candidate-live cosine
  (split by attention overlap), and two GT-free merge signals on live slots:
  low-q̃_1 attention (Perron residual) and within-slot DINO feature variance.

Usage (inside the container):
  python event_analysis/lcc_candidate_probe.py <config> <checkpoint> [n_batches]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from slotcurri import configuration, data, models

CFG = sys.argv[1]
CKPT = sys.argv[2]
N_BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MIN_PATCHES = 4
TAUS = (0.05, 0.1, 0.2)

config = configuration.load_config(CFG)
# q̃_1 (Perron mode of G_s) is only emitted with the Perron readout on.
config.model.attn_mass_curriculum.eval_perron_readout = True
dataset = data.build(config.dataset)
dataset.setup("validate")
loader = dataset.val_dataloader()

model = models.build(config.model, config.optimizer, None, None)
ckpt = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)
model = model.cuda().eval()

acc = {k: [] for k in [
    "n_live", "n_obj", "merge", "obj_cand_owner", "cand_att", "cand_att_bg_frac",
    "cand_att_on_merged", "best_cand_att_on_merged", "live_att_on_merged",
    "cos_cc", "cos_cl_max", "live_owner_objs", "cos_cc_overlap", "cos_cc_disjoint",
    "q_single", "q_merged_primary", "q_merged_secondary",
    "resid_single_slot", "resid_merged_slot",
]}
printed = False
with torch.no_grad():
    for bi, batch in enumerate(loader):
        if batch is None:
            continue
        if bi >= N_BATCHES:
            break
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model.forward(batch, train=False, cycle=False)
        proc = out["processor"]
        att = proc["state_attn_mask"].float()          # (B, T, S, F)
        pi = proc["active_mask"].float()               # (B, T, S)
        slots = proc["state"].float()                  # (B, T, S, D)
        dmask = out["decoder"]["masks"].float()        # (B, T, S, F)
        seg = batch["segmentations"]
        if not printed:
            print("att", tuple(att.shape), "pi", tuple(pi.shape), "dmask", tuple(dmask.shape),
                  "seg", tuple(seg.shape), seg.dtype)
            print("encoder keys", {k: tuple(v.shape) for k, v in out["encoder"].items()
                                   if torch.is_tensor(v)})
            printed = True
        feat = F.normalize(out["encoder"]["backbone_features"].float(), dim=-1)
        feat = feat.reshape(att.shape[0], att.shape[1], att.shape[-1], -1)  # (B, T, F, C)
        # attention-weighted within-slot variance, per unit attention mass
        w = att / att.sum(-1, keepdim=True).clamp_min(1e-8)
        mu = w @ feat                                           # (B, T, S, C)
        sq = (feat.square().sum(-1)).unsqueeze(2)               # (B, T, 1, F)
        var = (w * sq).sum(-1) - mu.square().sum(-1)            # (B, T, S)
        B, T, S, Fp = att.shape
        H = W = int(Fp ** 0.5)
        # seg -> (B, T, H, W) integer ids at patch resolution (nearest)
        if seg.ndim == 6:
            seg = seg.squeeze(2) if seg.shape[2] == 1 else seg.squeeze(-1)
        if seg.ndim == 5:  # (B, T, K, h, w) one-hot or (B, T, h, w, 1)
            if seg.shape[-1] == 1:
                ids = seg[..., 0].long()
            else:
                ids = seg.float().argmax(2)
        else:
            ids = seg.long()
        ids = F.interpolate(ids.float().flatten(0, 1).unsqueeze(1), size=(H, W),
                            mode="nearest").long().view(B, T, Fp)

        pmax = pi.amax(-1, keepdim=True).clamp_min(1e-8)
        pbar = pi / pmax
        live = pbar > 0.5
        owner = dmask.argmax(2)  # (B, T, F)
        dg = proc.get("decoder_gate")  # π q̃_1, (B*T?, S, F) or (B, T, S, F)
        qt = None
        if dg is not None:
            dg = dg.float().reshape(B, T, S, Fp)
            qt = dg / pi.unsqueeze(-1).clamp_min(1e-8)
        u = F.normalize(slots, dim=-1)
        cos = u @ u.transpose(-1, -2)

        for b in range(B):
            for t in range(T):
                lv = live[b, t]
                cd = ~lv
                acc["n_live"].append(lv.sum().item())
                a = att[b, t]  # (S, F)
                cand_att = a[cd].sum(0)  # (F,)
                idf = ids[b, t]
                bg = idf == 0
                acc["cand_att"].append(cand_att.mean().item())
                acc["cand_att_bg_frac"].append(
                    (cand_att[bg].sum() / cand_att.sum().clamp_min(1e-8)).item())
                if cd.sum() >= 2:
                    cc = cos[b, t][cd][:, cd]
                    iu = torch.triu(torch.ones_like(cc, dtype=torch.bool), 1)
                    acc["cos_cc"].append(cc[iu].mean().item())
                    # overlap_ij = sum_n min(A_i, A_j) / min(mass_i, mass_j)
                    ac = a[cd]
                    mass = ac.sum(-1).clamp_min(1e-6)
                    ov = torch.minimum(ac.unsqueeze(1), ac.unsqueeze(0)).sum(-1)
                    ov = ov / torch.minimum(mass.unsqueeze(1), mass.unsqueeze(0))
                    active = (mass.unsqueeze(1) > 5.0) & (mass.unsqueeze(0) > 5.0)
                    hi = iu & active & (ov > 0.5)
                    lo = iu & ~(active & (ov > 0.1))
                    acc["cos_cc_overlap"].extend(cc[hi].tolist())
                    acc["cos_cc_disjoint"].extend(cc[lo].tolist())
                if cd.any() and lv.any():
                    acc["cos_cl_max"].append(cos[b, t][cd][:, lv].amax(1).mean().item())
                objs = [k for k in idf.unique().tolist() if k != 0
                        and (idf == k).sum() >= MIN_PATCHES]
                if not objs:
                    continue
                acc["n_obj"].append(len(objs))
                own = {}
                for k in objs:
                    m = idf == k
                    s = dmask[b, t][:, m].sum(1).argmax().item()
                    own.setdefault(s, []).append(k)
                    acc["obj_cand_owner"].append(float(not lv[s].item()))
                for s, ks in own.items():
                    if lv[s]:
                        acc["live_owner_objs"].append(len(ks))
                    merged = len(ks) > 1
                    if lv[s]:
                        # object-only variance: restrict the slot's attention to fg of owned objects
                        m_own = torch.zeros_like(idf, dtype=torch.bool)
                        for k in ks:
                            m_own |= idf == k
                        ww = a[s] * m_own
                        ww = ww / ww.sum().clamp_min(1e-8)
                        f_bt = feat[b, t]
                        mu_o = ww @ f_bt
                        v_obj = ((ww * f_bt.square().sum(-1)).sum() - mu_o.square().sum()).item()
                        tag = "merged" if merged else "single"
                        acc.setdefault(f"var_full_{tag}", []).append(var[b, t, s].item())
                        acc.setdefault(f"var_obj_{tag}", []).append(v_obj)
                    if qt is not None and lv[s]:
                        q = qt[b, t, s]
                        # mean q̃ over each owned object, attention-weighted
                        qs = []
                        for k in ks:
                            m = idf == k
                            w = a[s, m]
                            qs.append((w * q[m]).sum().item() / max(w.sum().item(), 1e-8))
                        resid = ((a[s] * (1 - q)).sum() / a[s].sum().clamp_min(1e-8)).item()
                        tag = "merged" if merged else "single"
                        for tau in TAUS:
                            low = (q < tau).float()
                            acc.setdefault(f"lowq{tau}_{tag}_slot", []).append(
                                ((a[s] * low).sum() / a[s].sum().clamp_min(1e-8)).item())
                            if merged:
                                # recall: secondary objects' attention that falls below tau
                                prim = max(ks, key=lambda k: (a[s, idf == k] * q[idf == k]).sum().item())
                                sec = torch.zeros_like(low, dtype=torch.bool)
                                for k in ks:
                                    if k != prim:
                                        sec |= idf == k
                                acc.setdefault(f"lowq{tau}_secondary_recall", []).append(
                                    ((a[s] * low * sec).sum() / (a[s] * sec).sum().clamp_min(1e-8)).item())
                                acc.setdefault(f"lowq{tau}_precision_merged", []).append(
                                    ((a[s] * low * sec).sum() / (a[s] * low).sum().clamp_min(1e-8)).item())
                        if merged:
                            qs.sort(reverse=True)
                            acc["q_merged_primary"].append(qs[0])
                            acc["q_merged_secondary"].extend(qs[1:])
                            acc["resid_merged_slot"].append(resid)
                        else:
                            acc["q_single"].append(qs[0])
                            acc["resid_single_slot"].append(resid)
                    for k in ks:
                        acc["merge"].append(float(merged))
                        if merged:
                            m = idf == k
                            acc["cand_att_on_merged"].append(cand_att[m].mean().item())
                            if cd.any():
                                acc["best_cand_att_on_merged"].append(
                                    a[cd][:, m].mean(1).max().item())
                            acc["live_att_on_merged"].append(a[lv][:, m].sum(0).mean().item())

print("checkpoint:", CKPT)
print("num_slots:", S)
for k, v in acc.items():
    if v:
        print("  %-26s mean %.4f  (n=%d)" % (k, float(np.mean(v)), len(v)))
for k in ["cos_cc_overlap", "cos_cc_disjoint", "q_single", "q_merged_primary",
          "q_merged_secondary", "resid_single_slot", "resid_merged_slot",
          "var_full_single", "var_full_merged", "var_obj_single", "var_obj_merged"]:
    if acc[k]:
        q = np.percentile(acc[k], [10, 25, 50, 75, 90])
        print("  %-26s p10/25/50/75/90 %s" % (k, " ".join("%.3f" % x for x in q)))
