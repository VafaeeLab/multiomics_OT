#!/usr/bin/env python3
"""L-method elbow on exp(k*a) — the selected ambiguity threshold rule.

Production's rule is max-distance-to-chord on the RAW score curve, clamped to
[8%, 50%]. For pseudo_label and label_margin that curve is close to a straight
line, so the clamp decides the threshold in ~64% / ~62% of runs rather than the
data. This module replaces both halves of that:

  TRANSFORM   a -> exp(k*a), k=10. Strictly increasing, so the ambiguity
              RANKING of cells is identical to the original method - only the
              threshold moves. Computed as exp(k*(a-1)), which is the same
              curve scaled by the positive constant exp(-k); the elbow is
              invariant to that rescaling and it avoids overflow at large k.

  RULE        L-method: fit two least-squares lines to the sorted curve and
              split where the size-weighted RMSE is minimised. Asks "where does
              one line stop describing this", which stays well posed on a
              smooth curve. O(n) via prefix sums.

  NO CLAMP    the point is to let the rule answer on its own.

Why this pair, measured over 624 runs per detector:
  - no rule rescues the raw curve: chord 30%, L-method 33%, curvature 31%,
    Otsu 39%, GMM 34% land under the 50% cap. The curve is the problem.
  - on exp(10a) the threshold clears the cap in 90% / 93% of runs, flagging
    30.1% / 26.4% of cells.
  - chord and L-method are independent constructions; on raw they agree within
    2pp on only 35-43% of curves, on exp(10a) 77-82%. Two unrelated rules
    converging is the evidence that a real elbow exists.
  - transforms that flag under ~15% (logit, neglogm, odds) all lose any
    correlation with pair difficulty - they impose a fraction rather than find
    one, so a smaller cut is not automatically a better one.

NOT for local_consistency: ~71% of its scores are exactly zero, so its curve
already falls off a cliff at ~29% and exp collapses the threshold to 2-3%,
which failed downstream. That detector keeps its original rule.
"""
import numpy as np

K_DEFAULT = 10.0


def transform(scores, k=K_DEFAULT):
    """exp(k*a) as exp(k*(a-1)). Strictly increasing => ranking preserved."""
    a = np.clip(np.asarray(scores, float), 0.0, 1.0)
    return np.exp(k * (a - 1.0))


def lmethod_index(v):
    """Index of the two-line breakpoint on v (sorted DESCENDING).

    The residual sum of squares of a least-squares line over a contiguous run
    is Syy - Sxy^2/Sxx on centred sums, so with prefix sums every candidate
    split costs O(1) and the whole scan is O(n).
    """
    v = np.asarray(v, float)
    n = len(v)
    if n < 40 or not np.isfinite(v).all() or np.allclose(v, v[0]):
        return None
    x = np.arange(n, dtype=float)
    cx = np.concatenate([[0.0], np.cumsum(x)])
    cy = np.concatenate([[0.0], np.cumsum(v)])
    cxx = np.concatenate([[0.0], np.cumsum(x * x)])
    cyy = np.concatenate([[0.0], np.cumsum(v * v)])
    cxy = np.concatenate([[0.0], np.cumsum(x * v)])

    def rss(lo, hi):
        m = (hi - lo).astype(float)
        sx, sy = cx[hi] - cx[lo], cy[hi] - cy[lo]
        vxx = (cxx[hi] - cxx[lo]) - sx * sx / m
        vyy = (cyy[hi] - cyy[lo]) - sy * sy / m
        vxy = (cxy[hi] - cxy[lo]) - sx * sy / m
        return np.maximum(
            vyy - np.where(vxx > 1e-12, vxy * vxy / np.maximum(vxx, 1e-12), 0.0),
            0.0)

    i = np.arange(5, n - 5)
    tot = (i / n) * np.sqrt(rss(np.zeros_like(i), i) / np.maximum(i, 1)) + \
          ((n - i) / n) * np.sqrt(rss(i, np.full_like(i, n)) /
                                  np.maximum(n - i, 1))
    return int(i[int(np.argmin(tot))])


def chord_index(v):
    """Production's rule, kept for the comparison recorded in summary.json."""
    v = np.asarray(v, float)
    n = len(v)
    if n < 20 or not np.isfinite(v).all() or np.allclose(v, v[0]):
        return None
    x = np.arange(n, dtype=float)
    d = np.abs((v[-1] - v[0]) * x - (n - 1) * v + (n - 1) * v[0]) / \
        (np.hypot(v[-1] - v[0], n - 1) + 1e-12)
    return int(np.argmax(d))


MAX_FRAC = 0.50


def knee(scores, k=K_DEFAULT, label=""):
    """Cells to flag: L-method elbow on exp(k*a), capped at MAX_FRAC.

    The transform moves the elbow below half the cells in ~90% of runs, but not
    all: on the remainder it still lands past 50%, up to 93%. Removing most of
    a dataset is not a usable answer whatever the curve says, so the cap stands
    as a policy limit.

    When it binds, that fact is PRINTED rather than silently applied - a
    threshold set by the guardrail is a different object from one found in the
    data, and the earlier chord rule hid exactly this behind a clamp for two
    thirds of its runs. `label` should identify method / dataset / modality so
    the warning names the run it came from.

    Falls back to the chord elbow on the same transformed curve if the L-method
    cannot run (fewer than 40 finite scores, or a constant curve), and to 0 if
    neither applies - 0 is honest here, where a guess would not be.
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 20:
        return 0
    v = np.sort(transform(s, k))[::-1]
    i = lmethod_index(v)
    if i is None:
        i = chord_index(v)
    if not i:
        return 0
    cap = int(MAX_FRAC * n)
    if i > cap:
        print(f"  [WARNING] threshold past {MAX_FRAC:.0%}: {label} "
              f"elbow at {100.0 * i / n:.1f}% of {n} cells -> capped at "
              f"{100.0 * cap / n:.1f}% ({cap} cells)", flush=True)
        return cap
    return int(i)


def diagnostics(scores, k=K_DEFAULT):
    """Both rules on both curves, for the record written to summary.json.

    |chord - lmethod| on the transformed curve is the honesty check: two
    unrelated rules landing together means the elbow is real, and a large gap
    means neither found one and the number should not be trusted.
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 20:
        return {}
    raw = np.sort(s)[::-1]
    tr = np.sort(transform(s, k))[::-1]

    def pct(i):
        return None if i is None else round(100.0 * i / n, 2)

    lt, ct = lmethod_index(tr), chord_index(tr)
    out = {"k": k,
           "pct_lmethod_exp": pct(lt), "pct_chord_exp": pct(ct),
           "pct_lmethod_raw": pct(lmethod_index(raw)),
           "pct_chord_raw": pct(chord_index(raw))}
    if lt is not None and ct is not None:
        out["rule_gap_pp"] = round(abs(100.0 * (lt - ct) / n), 2)
        out["rules_agree_2pp"] = bool(out["rule_gap_pp"] < 2.0)
    return out
