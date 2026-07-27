# Attention-Mass Curriculum for Soft Slot Allocation in Video Object-Centric Learning

**One-line pitch.** Replace SlotCurri’s hard slot-count jumps with a continuous, attention-mass soft gate that grows capacity only where slots actually claim content.

**Status.** Draft for a SlotCurri (CVPR 2026) follow-up. Final YTVIS numbers are placeholders until v9/v10 finish 100k steps. Do **not** claim literature SOTA from mid-run leads.

---

## Abstract

Video object-centric models that reconstruct dense features with a fixed slot pool are biased toward using every slot, which often fragments single objects into redundant parts. SlotCurri mitigates this by a reconstruction-guided curriculum that starts with few slots and expands the pool in discrete stages. While effective, hard slot-count schedules impose abrupt capacity jumps, ignore per-frame object load, and still treat inactive slots as an all-or-nothing architectural choice.

We propose an **attention-mass curriculum**: each slot receives a differentiable soft gate \(g_s=\sigma((\mathrm{mass}_s-p)/\tau)\) from its spatial attention mass, which modulates state updates and decoder mixing. Annealing the threshold \(p\) implements a continuous coarse-to-fine capacity schedule without changing the slot count. In practice, stable training requires several design choices—fixed moderate temperature, log-scheduled \(p\), winner-normalized state mixing, and ungated temporal contrastive learning—so that gates shape allocation without collapsing into a few dominant slots or making contrastive alignment artificially easy.

On YouTube-VIS 2021, the soft curriculum expands effective capacity in lockstep with accuracy and is competitive with the SlotCurri hierarchical baseline **[TBD: final ARI / mBO vs SlotCurri]**. Ablations show that carrying the curriculum primarily with \(p\) under fixed \(\tau\) outperforms soft-to-hard temperature annealing **[TBD: v10 vs v9 finals]**.

---

## 1. Introduction

Object-centric learning aims to bind raw sensory streams to a small set of entity representations. In video, slot-attention architectures typically maintain a fixed pool of slots, propagate them over time, and reconstruct encoder features. Reconstruction alone, however, rewards covering every patch: unused slots become liabilities, and the model often splits one object across several slots (**over-fragmentation**). The result is brittle object identity, weaker segmentation metrics, and capacity that is spent on redundancy rather than new entities.

**SlotCurri** addresses this mismatch between pool size and scene complexity with a *slot curriculum*. Training begins with a coarse budget and expands the number of slots when reconstruction still fails, accompanied by a structure-aware (SSIM) loss that sharpens semantic boundaries and cyclic inference for temporal consistency. On real-world YouTube-VIS, this hard hierarchical schedule (e.g., \(2\rightarrow3\rightarrow7\) slots) substantially reduces fragmentation relative to a static pool.

Yet a hard count curriculum leaves three gaps. First, capacity changes are **discontinuous**: when the budget jumps, many new slots become fully live at once, even if the scene only needs a fractional increase. Second, allocation is **video-global and stage-global**, not conditioned on how much content each slot currently claims in a given frame. Third, dormant capacity is still an architectural on/off decision; under soft reconstruction objectives, there is little notion of a *partially* active slot that can grow as mass accumulates.

We therefore revisit the curriculum module of SlotCurri while keeping its structure-aware training and cyclic inference. Instead of growing the slot *count*, we keep a fixed pool and learn a **soft activity** for each slot from **attention mass**—the fraction of spatial features claimed by that slot’s attention. A logistic gate

\[
g_s = \sigma\!\left(\frac{\mathrm{mass}_s^{(\gamma)} - p}{\tau}\right)
\]

maps mass to \([0,1]\), optionally after \(\gamma\)-sharpening of per-patch attention to suppress accidental mass on always-second slots. Gates enter the temporal state update (slots with small \(g\) stay sticky) and the decoder mixture (attention weights reweighted by \(g\) and renormalized). Annealing \(p\) from a high to a low multiple of the uniform mass \(1/S\) yields a continuous coarse-to-fine schedule: early training favors a few high-mass slots; later training admits smaller claimants.

Naive soft gating is not automatically better than hard curricula. In early attempts, soft gates either collapsed onto one or two winners, or opened too uniformly when the temperature was wide—especially once state mixing and contrastive losses interacted with absolute gate magnitudes. Our working recipe treats **relative** gate structure as the quantity that matters for reconstruction (decoder renormalization; state updates normalized by \(\max_s g_s\)), keeps temporal contrastive learning on the full slot pool so weak slots still receive identity gradients, and lets a **log-annealed** \(p\) carry the curriculum under a **fixed** moderate \(\tau\). Empirically, this fixed-\(\tau\) variant expands capacity from a concentrated early distribution toward a broader soft allocation, whereas temperature annealing from wide to sharp tends to stay diffuse early and under-translate added capacity into accuracy.

**Contributions.**

1. **Attention-mass soft curriculum.** We replace SlotCurri’s discrete slot-count schedule with a differentiable per-slot, per-frame gate derived from attention mass, enabling continuous capacity growth without changing \(S\).

2. **Training recipe for stable soft allocation.** We identify failure modes of gated curricula (collapse, over-easy gated contrastive, absolute-scale sticky updates) and stabilize them with winner-normalized state mixing, ungated contrastive loss, \(\gamma\)-sharpened mass, and a \(p\)-driven schedule with fixed \(\tau\).

3. **Empirical study on YouTube-VIS.** Relative to the SlotCurri hierarchical baseline and prior soft-gate ablations, the proposed curriculum **[TBD: matches / improves ARI–mBO at 100k]** and exhibits capacity trajectories that co-move with validation accuracy, with gate distributions that evolve from concentrated to multi-slot soft allocation.

**Teaser.** Figure~X **[TBD]** plots validation ARI against effective active capacity (\(\sum_s g_s\)) for the hierarchical SlotCurri baseline, a soft-gate ablation with temperature annealing, and our fixed-\(\tau\) attention-mass curriculum. Table~1 **[TBD: final numbers]** summarizes YTVIS FG-ARI / ARI / mBO. Mid-training snapshots already show the soft curriculum tracking SlotCurri’s accuracy while avoiding discrete slot jumps; we freeze claims after full 100k schedules complete.

---

## Notes for later (not for submission text)

### Ablation map (internal)

| Run | Curriculum idea | Outcome (as of ~55k / finals where done) |
|---|---|---|
| SlotCurri baseline | Hard hier. \(2\rightarrow3\rightarrow7\) | Final ARI 0.436 / mBO 0.345 @100k |
| v7 | Soft gate, forward \(\tau\) anneal, \(\gamma=2\) | Final ARI 0.408; late over-open (\(\sum g\sim6.5\)) |
| v8 | Reverse \(\tau\) | Final ARI 0.392; visually softer, weaker ARI |
| v9 | + `p_start=1.5`, log \(p\), `state_max_norm`, ungated contrastive, \(\tau:1\rightarrow0.3\) | ~55k ARI ~0.39; early soft-uniform gates; capacity↑ but slower ARI↑ |
| **v10 (main)** | v9 package but **fixed \(\tau=0.3\)** | ~55k ARI 0.44 (best peak 0.456 @50k); early concentrated → later multi-slot soft |

### Metrics to freeze at 100k

- `val/ari`, `val/mbo`, `val/image_ari`, `val/image_mbo`
- `train/active_slots` (\(=\sum g\)), `train/gate_max`, `train/gate_top2`, `train/gate_entropy`, `train/gate_n_half`
- Gate distribution at early / mid / late checkpoints (top2 share, \(H/\log S\), \(\sum g/\max g\))
- Optional: per-frame temporal stability of \(g\) (argmax persistence, \(\|\Delta\mathrm{share}\|_1\))

### Figures to draw

1. Method diagram: mass → \(g\) → state mix / decoder renorm; \(p\) schedule.
2. Active-capacity curves vs step (baseline slot count stages overlaid).
3. Gate distribution evolution (v9 vs v10): top2%, entropy, \(n_{g>0.5}\).
4. Accuracy vs capacity scatter / dual-axis plot.
5. Qualitative YTVIS masks: baseline vs v10 (and failure cases).

### Wording constraints

- Frame as **improving SlotCurri’s curriculum module**, not replacing the whole paper.
- Mid-run leads: “on track / competitive,” not “SOTA.”
- Literature comparisons beyond SlotCurri only after a proper related-work pass and final tables.

### Placeholder block for Abstract/Intro once runs finish

```
[TBD: final ARI / mBO vs SlotCurri]
[TBD: v10 vs v9 finals]
[TBD: Table 1 YTVIS numbers]
[TBD: Figure X teaser]
```

Preliminary (not for paper body until confirmed):

- SlotCurri baseline @100k: ARI 0.4359, mBO 0.3445
- v10 @55k: ARI 0.4433 (best 0.4563 @50k), mBO 0.3282
- v9 @55k: ARI 0.3874 (best 0.4086 @47.5k), mBO 0.3148
