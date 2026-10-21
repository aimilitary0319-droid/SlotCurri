"""Implementation check for feature_curriculum.xvid_energy (true / frame / balance) and xvid_power.

CPU only. Synthetic tokens x = shared + per-video + per-object + noise, then
the real NcutRelationalLeveling / FrameEncoder / MapOverTime path.
"""
import torch
from torch import nn

from slotcurri.modules.encoders import FrameEncoder, NcutRelationalLeveling
from slotcurri.modules.video import MapOverTime

torch.manual_seed(0)
B, T, N, D = 6, 4, 64, 32


def make_tokens():
    shared = 3.0 * torch.randn(1, 1, 1, D)
    video = 1.0 * torch.randn(B, 1, 1, D)
    centers = 1.5 * torch.randn(B, 1, 3, D)
    assign = torch.randint(0, 3, (B, T, N))
    obj = torch.gather(
        centers.expand(B, T, 3, D), 2, assign.unsqueeze(-1).expand(B, T, N, D)
    )
    return shared + video + obj + 0.5 * torch.randn(B, T, N, D)


def energy(t, mu):
    return (t - mu).square().sum(-1).mean().item()


def rel_err(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


results = []
x4 = make_tokens()
x = x4.flatten(0, 1)  # (B*T, N, D), what MapOverTime hands the encoder
mu = x.mean(dim=(0, 1), keepdim=True)

plain = NcutRelationalLeveling(chunk_size=8, barrier=False, xvid_energy=False)
px = plain._level(x)
m_f = px.mean(dim=1, keepdim=True)

# --- mode parsing ---
modes = {v: NcutRelationalLeveling(barrier=False, xvid_energy=v).xvid_mode
         for v in [False, True, "true", "false", "frame", "FRAME", "none", "balance", "video"]}
results.append(check("mode parsing", modes == {
    False: "off", True: "all", "true": "all", "false": "off",
    "frame": "frame", "FRAME": "frame", "none": "off", "balance": "balance",
    "video": "video"}, str(modes)))
for bad in [dict(xvid_energy="bogus"), dict(xvid_energy="split"),
            dict(xvid_energy=True, xvid_power=1.5), dict(xvid_energy=True, xvid_power=-0.1),
            dict(xvid_energy="frame", xvid_power=0.5), dict(xvid_energy=False, xvid_power=0.5),
            dict(xvid_energy="balance", xvid_power=0.5)]:
    try:
        NcutRelationalLeveling(barrier=False, **bad)
        results.append(check(f"rejected {bad}", False))
    except ValueError:
        results.append(check(f"rejected {bad}", True))

# --- off == old path ---
results.append(check("off mode equals plain P X",
                     torch.equal(plain._level(x), px)))

g_full = (energy(x, mu) / energy(px, mu)) ** 0.5
for beta in [0.0, 0.5, 0.75]:
    mod = NcutRelationalLeveling(chunk_size=8, barrier=False, xvid_energy=True,
                                 xvid_power=beta)
    out = mod._level(x)
    g = mod.last_xvid_gain
    results.append(check(f"[power {beta}] gain == g^beta",
                         rel_err(g, g_full ** beta) < 1e-5,
                         f"g^beta {g:.5f} ref {g_full ** beta:.5f}"))
    results.append(check(f"[power {beta}] out == mu + g^beta (PX - mu)",
                         torch.allclose(out, mu + g * (px - mu), atol=1e-5)))
    results.append(check(f"[power {beta}] energy around mu == g^(2 beta) E|PX-mu|^2",
                         rel_err(energy(out, mu), g ** 2 * energy(px, mu)) < 1e-4))
results.append(check("[power 0] equals plain P X",
                     torch.allclose(NcutRelationalLeveling(
                         chunk_size=8, barrier=False, xvid_energy=True,
                         xvid_power=0.0)._level(x), px, atol=1e-6)))
full = NcutRelationalLeveling(chunk_size=8, barrier=False, xvid_energy=True)._level(x)
results.append(check("[power 1] bit-identical to default xvid_energy=true",
                     torch.equal(NcutRelationalLeveling(
                         chunk_size=8, barrier=False, xvid_energy=True,
                         xvid_power=1.0)._level(x), full)))


# --- balance: the shared scale moves g_bal below 1, inside (1, g_full), above g_full ---
def centered(t):
    return t - t.mean(-1, keepdim=True)


seen = set()
for scale in [0.0, 0.5, 1.0, 2.0, 3.0, 6.0]:
    xs = (x - mu) + scale * mu
    mu_s = xs.mean(dim=(0, 1), keepdim=True)
    px_s = plain._level(xs)
    gf = (energy(xs, mu_s) / energy(px_s, mu_s)) ** 0.5
    gb = (centered(mu_s).square().sum().item()
          / centered(px_s - mu_s).square().sum(-1).mean().item()) ** 0.5
    case = "low" if gb < 1.0 else ("high" if gb > gf else "inside")
    seen.add(case)
    mod = NcutRelationalLeveling(chunk_size=8, barrier=False, xvid_energy="balance")
    out = mod._level(xs)
    g = mod.last_xvid_gain
    g_ref = min(max(gb, 1.0), gf)
    tag = f"[balance x{scale} {case}]"
    results.append(check(f"{tag} g == clamp(g_bal, 1, g_full)", rel_err(g, g_ref) < 1e-4,
                         f"g {g:.4f}  g_bal {gb:.4f}  g_full {gf:.4f}"))
    results.append(check(f"{tag} out == mu + g (PX - mu)",
                         torch.allclose(out, mu_s + g * (px_s - mu_s), atol=1e-4)))
    if case == "inside":
        e_shared = centered(mu_s).square().sum().item()
        e_rest = centered(out - mu_s).square().sum(-1).mean().item()
        results.append(check(f"{tag} centered shared energy == centered rest",
                             rel_err(e_rest, e_shared) < 1e-4,
                             f"shared {e_shared:.4f} rest {e_rest:.4f}"))
results.append(check("[balance] all three clamp cases exercised",
                     seen == {"low", "inside", "high"}, str(sorted(seen))))

for mode in ["all", "frame", "power0.5", "balance"]:
    xv = {"frame": "frame", "balance": "balance"}.get(mode, True)
    power = 0.5 if mode == "power0.5" else 1.0
    mod = NcutRelationalLeveling(chunk_size=8, barrier=False, xvid_energy=xv,
                                 xvid_power=power)
    out = mod._level(x)
    g = mod.last_xvid_gain
    e_raw, e_px, e_out = energy(x, mu), energy(px, mu), energy(out, mu)
    if mode in ("all", "frame"):
        results.append(check(f"[{mode}] energy around mu matches raw X",
                             rel_err(e_out, e_raw) < 1e-4,
                             f"raw {e_raw:.4f}  PX {e_px:.4f}  out {e_out:.4f}  g {g:.4f}"))
    if mode == "all":
        expect = mu + g * (px - mu)
        results.append(check("[all] out == mu + g (PX - mu)",
                             torch.allclose(out, expect, atol=1e-5)))
        g_ref = (energy(x, mu) / energy(px, mu)) ** 0.5
        results.append(check("[all] g == sqrt(E|X-mu|^2 / E|PX-mu|^2)",
                             rel_err(g, g_ref) < 1e-5, f"g {g:.5f} ref {g_ref:.5f}"))
    elif mode == "frame":
        resid_out = out - out.mean(dim=1, keepdim=True)
        results.append(check("[frame] within-frame residual unchanged (== PX - m_f)",
                             torch.allclose(resid_out, px - m_f, atol=1e-5)))
        results.append(check("[frame] frame mean == mu + g (m_f - mu)",
                             torch.allclose(out.mean(dim=1, keepdim=True),
                                            mu + g * (m_f - mu), atol=1e-5)))

    # chunk invariance: rescale uses the whole forward batch, not one chunk
    outs = []
    for cs in [1, 3, 8, 64]:
        m2 = NcutRelationalLeveling(chunk_size=cs, barrier=False, xvid_energy=xv,
                                    xvid_power=power)
        outs.append(m2._level(x))
    diff = max((o - outs[0]).abs().max().item() for o in outs)
    results.append(check(f"[{mode}] chunk_size invariant (1,3,8,64)", diff < 1e-5,
                         f"max diff {diff:.2e}"))

    # mix convention
    f0, f1, f3 = mod(x, 0.0), mod(x, 1.0), mod(x, 0.3)
    results.append(check(f"[{mode}] mix 0 == rescaled, 1 == raw, 0.3 == blend",
                         torch.allclose(f0, out, atol=1e-6) and f1 is x
                         and torch.allclose(f3, 0.7 * out + 0.3 * x, atol=1e-5)))

    # half input: computed in fp32, returned in input dtype
    oh = mod._level(x.half())
    results.append(check(f"[{mode}] fp16 in -> fp16 out, finite, close to fp32",
                         oh.dtype == torch.float16 and torch.isfinite(oh).all().item()
                         and (oh.float() - out).abs().max().item() < 0.05,
                         f"max diff {(oh.float() - out).abs().max().item():.3e}"))

    # no gradient into the leveled branch
    xg = x.clone().requires_grad_(True)
    y = mod(xg, 0.3)
    y.sum().backward()
    results.append(check(f"[{mode}] grad only through the mix*x branch",
                         torch.allclose(xg.grad, torch.full_like(x, 0.3))))

# --- integration: MapOverTime(FrameEncoder) with apply=key ---
class IdentityBackbone(nn.Module):
    def forward(self, t):
        return t


fe = FrameEncoder(backbone=IdentityBackbone())
fe.feature_ncut = NcutRelationalLeveling(chunk_size=8, barrier=False, xvid_energy=True)
fe.feature_ncut_mix = 0.0
fe.feature_curriculum_apply = "key"
enc = MapOverTime(fe)

enc.train()
o = enc(x4)
ref = NcutRelationalLeveling(chunk_size=8, barrier=False, xvid_energy=True)._level(x)
results.append(check("[encoder] train: backbone_key == module on flattened (B*T,N,D)",
                     torch.allclose(o["backbone_key"], ref.unflatten(0, (B, T)), atol=1e-6)))
results.append(check("[encoder] train: recon target / Value stay raw X",
                     torch.equal(o["backbone_features"], x4)))

# If the module had seen (B, T, N, D), mu would be a per-patch-position mean.
px4 = px.unflatten(0, (B, T))
mu_pp = x4.mean(dim=(0, 1), keepdim=True)
g_pp = (energy(x4, mu_pp) / energy(px4, mu_pp)) ** 0.5
wrong = mu_pp + g_pp * (px4 - mu_pp)
gap = (o["backbone_key"] - wrong).abs().max().item()
results.append(check("[encoder] mu is the all-token mean, not per patch position",
                     gap > 1e-2, f"max |out - per-patch version| {gap:.3f}"))

enc.eval()
oe = enc(x4)
results.append(check("[encoder] eval: no backbone_key, features from raw X",
                     "backbone_key" not in oe and torch.equal(oe["backbone_features"], x4)))

print()
print(f"{sum(results)}/{len(results)} checks passed")
