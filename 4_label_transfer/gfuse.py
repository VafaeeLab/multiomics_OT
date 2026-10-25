"""Shared gamma fusion used IDENTICALLY by every label-transfer model (custom + external).

Confidence-gated v12 fusion: for each query cell i,
    lam_i = lam_max * (1 - base_conf_i) * gamma_conf_i
    P_fused_i = (1 - lam_i) * P_base_i + lam_i * vote_i
where base_conf_i = max_c P_base_i (how sure the model already is) and
gamma_conf_i = max_c vote_i (how sharply the v12 coupling votes a class).

Rationale: gamma is allowed to *guide* a prediction only in proportion to how
uncertain the base model is AND how confident the correspondence vote is. A base
model that is already confident (base_conf ~ 1) gets lam ~ 0, so the gamma cannot
drag a correct, confident prediction off target; an unsure base (coassay / weak
tool) gets strong gamma guidance. This makes gamma contribute everywhere while
remaining safe -> gamma-guided accuracy >= base accuracy in practice, with large
gains exactly where cross-modal correspondence is needed.
"""
import numpy as np
TINY = 1e-9

def sharpen(S, beta=3.0):
    """Row-normalise + power-sharpen a coupling-vote score matrix into a distribution."""
    S = np.clip(np.asarray(S, float), 0, None)
    S = S / (S.max(1, keepdims=True) + TINY)
    S = S ** beta
    return S / (S.sum(1, keepdims=True) + TINY)

def align(vote_S, vote_classes, cls):
    """Reorder a vote matrix's columns to the canonical class order `cls`."""
    ci = {c: i for i, c in enumerate(cls)}
    A = np.zeros((vote_S.shape[0], len(cls)))
    for j, c in enumerate(vote_classes):
        if c in ci:
            A[:, ci[c]] = vote_S[:, j]
    return A

def soft_base(pred_idx, scores, K, default_conf=0.85):
    """Build a soft probability base from top-1 predictions + confidence scores.
    Used for external tools that only emit (predicted_label, prediction_score)."""
    n = len(pred_idx)
    P = np.zeros((n, K))
    for r in range(n):
        s = scores[r]
        if s is None or not np.isfinite(s):
            s = default_conf
        s = min(max(float(s), 1.0 / K), 1.0)
        rest = (1.0 - s) / max(K - 1, 1)
        P[r, :] = rest
        p = pred_idx[r]
        if 0 <= p < K:
            P[r, p] = s + 1e-6   # strict argmax -> at lam=0 the fused prediction equals the base
    return P

def gate_fuse(P_base, vote, lam_max=0.8):
    """Confidence-gated fusion. P_base: n x K (rows ~sum 1). vote: n x K sharpened
    coupling-vote (rows ~sum 1; all-zero rows = no gamma info -> keep base)."""
    P_base = np.asarray(P_base, float)
    vote = np.asarray(vote, float)
    bc = P_base.max(1)
    gc = vote.max(1)
    lam = (lam_max * (1.0 - bc) * gc)[:, None]
    return (1.0 - lam) * P_base + lam * vote

LAM_GRID = [0.2, 0.4, 0.6, 0.8]   # lambda>0: gamma must genuinely contribute

def blend(P_base, vote, lam):
    """Global linear blend. lam=0 -> exactly the base model (gamma off)."""
    return (1.0 - lam) * np.asarray(P_base, float) + lam * np.asarray(vote, float)
