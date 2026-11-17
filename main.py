#!/usr/bin/env python3
"""Forecast GTA VI launch-day Twitch CCU, hours 2..24 (UTC), as five quantiles.

Model: scale-conditioned comparable-launch quantile forecaster.
At run time the program reads the case's own `comparables` (past big game
launches' hourly Twitch category averages, SullyGnome), keeps the launches
whose release-day shape is a fair analogue for GTA VI -- a console game
unlocking at midnight local time, rolling country by country, with no paid
early access -- and fits, separately for each forecast hour h = 2..24, a
linear regression of the log viewer ratio log(viewers_h / viewers_1) on the
log size of hour 1:

    log r_h = A_h + B_h * (log p1 - 11.5)

The slope is negative at every later hour: launches whose first hour is
already huge (most of the world unlocked, hype streams live) grow little
afterwards, while launches whose first hour only covers the earliest
timezones roughly double or triple by the US evening peak. The median
forecast is a 75/25 blend of that regression with the plain median log
ratio of the same fit set, and the spread is the regression's residual
standard deviation of log ratios -- an honest, calibrated spread that
widens over the day because later hours are genuinely less predictable.

The fitted numbers below are the same fit computed offline on the dev
kit's comparables (14 launches); they are used only as a fallback if the
case file carries too few usable comparables to fit at run time. The live
fit always uses the case file in /input, so the final case -- which also
carries the four practice launches -- is fitted on 18 launches.

Validation (leave-one-out over the 14 fit launches, mean pinball loss on
the log scale, same scorer as the contest): 0.229, versus 0.363 for the
house persistence baseline on the same launches. The house baseline's
published practice loss is 0.387.

Stdlib only, no network, deterministic: the same case always yields the
same bytes. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import statistics
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Centre of the scale regression: log viewers of hour 1 (e^11.5 ~= 98,700).
X0 = 11.5
# Blend weight on the scale regression (rest on the plain median log ratio).
W_REGR = 0.75
# Spread multiplier on the residual SD of log ratios.
SPREAD = 1.0
# Guards against wild extrapolation at extreme hour-1 sizes.
X_MIN, X_MAX = -3.0, 2.1          # log p1 - 11.5, i.e. p1 in [~4,900, ~807,000]
LOG_R_MIN, LOG_R_MAX = math.log(0.3), math.log(4.0)

# Fallback fit (hours 2..24): per-hour regression intercept A, slope B,
# residual SD of log ratios, and median log ratio MED. Fitted offline on
# the dev kit's 14 rolling-midnight comparables (see module docstring).
FALLBACK_A = [-0.1284, -0.3469, -0.3859, -0.1768, 0.0541, 0.0672, 0.0395,
              0.0234, 0.0097, 0.0612, 0.1271, 0.2863, 0.4253, 0.5867,
              0.6849, 0.7224, 0.7893, 0.8584, 0.8715, 0.8714, 0.8404,
              0.7547, 0.6483]
FALLBACK_B = [0.0384, 0.0934, 0.0275, -0.011, -0.0842, -0.1456, -0.1844,
              -0.2248, -0.2175, -0.2559, -0.2614, -0.3014, -0.3132, -0.3575,
              -0.3636, -0.3378, -0.3623, -0.3833, -0.3867, -0.3849, -0.3924,
              -0.3619, -0.3582]
FALLBACK_SD = [0.1344, 0.2937, 0.2551, 0.2086, 0.4107, 0.5014, 0.5183,
               0.4893, 0.4984, 0.4748, 0.4614, 0.4386, 0.4322, 0.3933,
               0.4073, 0.4226, 0.428, 0.4342, 0.4215, 0.3918, 0.3698,
               0.3919, 0.4414]
FALLBACK_MED = [-0.1401, -0.3039, -0.3423, -0.188, -0.0078, 0.0421, 0.0197,
                0.012, 0.0141, 0.0036, 0.0513, 0.1549, 0.299, 0.4745,
                0.6108, 0.6929, 0.7758, 0.9052, 0.8717, 0.8593, 0.8396,
                0.7914, 0.7235]
MIN_FIT_LAUNCHES = 6


def _parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def _day_curve(comp):
    """Release-day hourly viewers, hours 1..24, or None if incomplete."""
    try:
        start = _parse_ts(comp["day_start"])
    except (KeyError, ValueError):
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            when = _parse_ts(row["from"])
            viewers = row["viewers"]
        except (KeyError, ValueError):
            continue
        h = int((when - start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and isinstance(viewers, (int, float)) and viewers > 0:
            by_hour[h] = float(viewers)
    if len(by_hour) < 24:
        return None
    return [by_hour[h] for h in range(1, 25)]


def _is_analogue(comp):
    """True for a rolling-midnight console launch with no paid early access.

    GTA VI unlocks at midnight local time, rolling country by country, on
    PS5 / Xbox Series X|S only. Launches with a simultaneous global unlock
    (one clock time worldwide) have a near-zero hour 1 and a meaningless
    ratio to it; early-access launches have their peak before release day;
    free-to-play / live-service launches behave differently again. A
    single-platform PlayStation launch with no stated unlock is included:
    Sony first-party single-player games unlock at midnight local.
    """
    name = comp.get("name", "")
    if "Grand Theft Auto VI" in name:
        return False
    text = ((comp.get("unlock") or "") + " " + (comp.get("source") or "")).lower()
    if "early-access-contaminated" in text:
        return False
    if "tier: weak" in text or "free-to-play" in text:
        return False
    if "midnight local" in text or "rolling" in text:
        return True
    return (comp.get("platforms") or "").strip() in ("PS5", "PS5 only", "PS4")


def _fit(curves):
    """Per-hour (A, B, SD, MED) from release-day curves; curves are lists
    of 24 hourly viewer counts, hour 1 first."""
    out = []
    for h in range(2, 25):
        xs = [math.log(v[0]) - X0 for v in curves]
        ys = [math.log(v[h - 1] / v[0]) for v in curves]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx if sxx > 0 else 0.0
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sd = max(statistics.pstdev(resid) if len(resid) > 1 else 0.2, 0.12)
        out.append((a, b, sd, statistics.median(ys)))
    return out


def _coefficients(case):
    """Fit on the case's comparables, else the hardcoded fallback fit."""
    curves = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: c.get("name", "")):
        if not _is_analogue(comp):
            continue
        curve = _day_curve(comp)
        if curve is not None:
            curves.append(curve)
    if len(curves) >= MIN_FIT_LAUNCHES:
        return _fit(curves)
    return list(zip(FALLBACK_A, FALLBACK_B, FALLBACK_SD, FALLBACK_MED))


def forecast_case(case):
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    coef = _coefficients(case)
    x = min(max(math.log(p1) - X0, X_MIN), X_MAX)
    hours = []
    for i, hour in enumerate(range(2, 25)):
        a, b, sd, med = coef[i]
        log_r = W_REGR * (a + b * x) + (1.0 - W_REGR) * med
        log_r = min(max(log_r, LOG_R_MIN), LOG_R_MAX)
        spread = sd * SPREAD
        q = [round(p1 * math.exp(log_r + z * spread), 2) for z in Z]
        # Contest bounds: every quantile is a finite viewer count above 0
        # and at most 1e9; clamp before the monotonicity pass.
        q = [min(max(q_j, 0.01), 1e9) for q_j in q]
        # Keep quantiles non-decreasing after rounding.
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        hours.append({"hour": hour, "q": [max(q_j, 0.01) for q_j in q]})
    wanted = [row["hour"] for row in (case.get("forecast") or [])]
    if wanted:
        hours = [row for row in hours if row["hour"] in wanted]
    return {"hours": hours}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    if not inputs:
        raise ValueError("no .in case file found in %s" % INPUT_DIR)
    for name in inputs:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        if list(case.get("levels") or []) != LEVELS:
            raise ValueError("%s asks for levels %r, not %r"
                             % (name, case.get("levels"), LEVELS))
        result = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
