#!/usr/bin/env python3
# polymarket-faa-attribution-02 entry (Python 3.13, standard library only).
#
# Forecast: for each market of Polymarket's FAA outage attribution family,
# the Yes price at every minute of the 240 minutes after the checkpoint, as
# five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that minute's price.
#
# Method (measured on the six practice cases with a local scorer built from
# Polymarket's public price history; the scorer reproduces the house
# baseline's published practice loss of 0.363589 exactly, and the practice
# cases are the NVIDIA event's, re-cut byte for byte):
#
#   * Persistence median: each market's price stays at its checkpoint price.
#     The pre-checkpoint returns do NOT predict the next 240 minutes here:
#     they are negatively correlated with the realized drift (mean reversion),
#     and every drift term tried on the practice made the loss worse, so no
#     drift.
#   * Spread: a random-walk spread, sigma * sqrt(h), where sigma is the
#     per-market standard deviation of one-minute price changes over the
#     last 120 minutes, winsorized at +-0.01. Winsorizing matters: a single
#     one-minute jump (Iran 0.49 -> 0.275; AI escape 0.215 -> 0.02 in the
#     family's first minutes) otherwise inflates the spread by an order of
#     magnitude, and long windows are poisoned by old regimes (markets that
#     opened near 0.5 and decayed). Floored at 0.0003, capped at 0.02.
#
#   * NO cross-market normalization. The sibling NVIDIA entry normalized the
#     medians halfway toward summing to 1.0, because its markets were
#     mutually exclusive CEO candidates. The FAA family's four hypotheses
#     are NOT mutually exclusive: Iran IS a foreign actor, a hack can be
#     done by Iran, and an AI escape is independent — so their Yes prices
#     have no reason to sum to 1, in either direction. Live prices confirm
#     it: at 2026-11-14T02:40Z the four Yes prices summed to 0.092 (Iran
#     0.017, foreign actor 0.0295, hack 0.0395, AI escape 0.006). Pulling
#     the medians toward a sum of 1.0 would multiply every forecast by
#     ~6x. Dropping the normalization costs little on practice (0.279336
#     vs 0.271567 with it) and avoids a structurally wrong prior on the
#     final.
#
#   * Short history is a non-issue for this method: the sigma window is
#     only the last 120 minutes, and truncating the practice series to the
#     last 960 minutes (~16h, what h00 will hold), 180 or even 30 minutes
#     leaves the practice loss unchanged (0.279336 / 0.279336 / 0.279341).
#     A market with fewer than 2 price changes falls back to the sigma
#     floor.
#
# Practice loss (local scorer, mean pinball loss in percentage points):
#   this entry 0.279336 vs house baseline 0.363589 (-23.2%), better on all
#   six practice cases (p00 0.4701, p01 0.3107, p02 0.3492, p03 0.2654,
#   p04 0.2047, p05 0.1362 vs baseline 0.5361, 0.3804, 0.4084, 0.3686,
#   0.2988, 0.2382). Median error 0.767503 (same medians as baseline),
#   90% coverage 0.877311. Variants tried and rejected on the same scorer:
#   normalization halfway to 1.0 (0.271567 on practice — rejected for the
#   structural reason above, not for its practice score), full normalization
#   (0.277890), sigma windows 60/90/180/360m (0.281020 / 0.280837 /
#   0.283891 / 0.284406), winsor caps +-0.005 / +-0.02 (0.284265 /
#   0.283828), sigma floor 0.0005 / cap 0.03 (0.280612), floor 0.0001 /
#   cap 0.01 (0.280236). The optimum is flat around 90-120 minutes and
#   +-0.01 winsorization; nothing else came close to the normalization
#   variants, which are rejected on structure, not on score.
#
# The program is deterministic: no network, no clock, no randomness. Two
# runs on the same case write byte-identical output.
#
# Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# (reads $MAKEFUN_INPUT_DIR, else /input; writes $MAKEFUN_OUTPUT_DIR, else
# /output; one <id>.out per <id>.in, in name order).

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level, to 6 decimals (as the
# house baseline uses them).
Z_TABLE = {0.05: -1.644854, 0.25: -0.67449, 0.5: 0.0, 0.75: 0.67449,
           0.95: 1.644854}
# One-minute changes are winsorized at +-WINSOR before the sigma estimate:
# a single jump must not set a market's spread for the next four hours.
WINSOR = 0.01
# How many minutes of one-minute changes feed the sigma estimate.
SIGMA_WINDOW = 120
# Floor and cap on the per-minute sigma, in price units.
SIGMA_FLOOR = 0.0003
SIGMA_CAP = 0.02


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation); only used for
    levels outside the fixed table, so the five scored levels stay exact."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("quantile level out of (0, 1): %r" % p)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def z_for(tau):
    return Z_TABLE.get(tau, norm_ppf(tau))


def sigma_winsorized(prices):
    """Per-minute sigma: sample std of the one-minute price changes over
    the last SIGMA_WINDOW minutes, each change winsorized to +-WINSOR.
    A series too short to estimate from falls back to the floor."""
    n = len(prices)
    if n < 2:
        return SIGMA_FLOOR
    w = prices[-(SIGMA_WINDOW + 1):] if n > SIGMA_WINDOW + 1 else prices
    d = [w[i + 1] - w[i] for i in range(len(w) - 1)]
    d = [max(-WINSOR, min(WINSOR, x)) for x in d]
    m = len(d)
    if m < 2:
        return SIGMA_FLOOR
    # A window with no variation at all (every one-minute change the same,
    # in particular all zero for a flat market) has sample variance exactly
    # zero, so the floor applies. Say so directly instead of computing a
    # zero variance through the (m - 1) division.
    if all(x == d[0] for x in d):
        return SIGMA_FLOOR
    mu = sum(d) / m
    var = sum((x - mu) * (x - mu) for x in d) / (m - 1)
    sig = math.sqrt(var)
    return max(SIGMA_FLOOR, min(SIGMA_CAP, sig))


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def forecast_case(case):
    horizon = case["horizon_minutes"]
    taus = case["quantiles"]
    zs = [z_for(t) for t in taus]
    markets = case["markets"]
    out = []
    for m in markets:
        prices = m["prices"]
        if not prices:
            raise ValueError("market %s has no prices" % m["id"])
        x = prices[-1]
        sig = sigma_winsorized(prices)
        rows = []
        for h in range(1, horizon + 1):
            sp = sig * math.sqrt(h)
            row = []
            prev = 0.0
            for z in zs:
                v = price6(x + z * sp)
                if v < prev:
                    v = prev
                row.append(v)
                prev = v
            rows.append(row)
        out.append({"id": m["id"], "q": rows})
    return {"forecasts": out}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        cid = fname[:-len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        fc = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w", encoding="utf-8") as f:
            json.dump(fc, f)
            f.write("\n")


if __name__ == "__main__":
    main()
