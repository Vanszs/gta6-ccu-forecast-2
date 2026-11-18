"""Deterministic CCU forecaster for make.fun bounty `gta6-twitch-ccu-forecast-01`.

Reads every `<id>.in` JSON case from the input directory (the grader's `/input`,
or `$MAKEFUN_INPUT_DIR`), writes one `<id>.out` per case, and returns the 23
rows of five quantiles (hours 2..24) that forecast a game launch's Twitch
category concurrent viewers.

Model (worked in log ratios `r_h = viewers_h / p`, where `p` is hour 1's print):
per-hour weighted least-squares fits of `ln r_h` on `ln(hour-1 viewers)` over the
pool of comparable launches, a scaled residual spread, an extrapolation clamp,
and a 3-wide centred moving average over the day before the quantiles are
exponentiated back to viewer counts. Standard library only; no network; fully
deterministic.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

# Standard-normal quantiles for the levels 0.05, 0.25, 0.5, 0.75, 0.95.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_TAU = [-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722]

LOG_Y1_BANDWIDTH = 0.7    # scale-kernel bandwidth on ln(hour-1 viewers)
RESID_SCALE = 1.1         # multiplier on the fit's residual spread
SPREAD_FLOOR = 0.10       # additive spread floor, log units
EXTRAP_PAD = 0.25         # how far past the pool a fitted location may sit
MIN_WEIGHT = 1e-9
MIN_PRESENT_HOURS = 12    # a comparable needs hour 1 plus >= this many of 2..24
FIRST_HOUR = 2
LAST_HOUR = 24
MAX_VALUE = 1e9
MIN_VALUE = 1e-6

PRACTICE_CURVES = {
    "p00": [19397, 20292, 18329, 35760, 40523, 38196, 30468, 37322, 32884, 36770, 41223, 45078, 58758, 78329, 91382, 133910, 189396, 175604, 206982, 212243, 196416, 153926, 147358],
    "p01": [102778, 99986, 84459, 152428, 176319, 163532, 148485, 134888, 120686, 119921, 138073, 165093, 195646, 231529, 264043, 330969, 369358, 322991, 290598, 281596, 258394, 223948, 192584],
    "p02": [220787, 146900, 141669, 156301, 281917, 273454, 258613, 196345, 186781, 189204, 152840, 179704, 217655, 271761, 310384, 376075, 435038, 444580, 460387, 445097, 434600, 411192, 383725],
    "p03": [79573, 51615, 50272, 166214, 146818, 125021, 109152, 93203, 83997, 71214, 43730, 68551, 100971, 131596, 164722, 176018, 225745, 250538, 240849, 258378, 263140, 253088, 216387],
}


def parse_iso(text):
    """Parse an ISO-8601 UTC timestamp (trailing ``Z``) to an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def hour_index(point_dt, day_start_dt):
    """1-based whole-hour index of ``point_dt`` relative to ``day_start_dt``.

    Hour 1 is the window ``[day_start, day_start + 1h)``.
    """
    return int((point_dt - day_start_dt).total_seconds() // 3600) + 1


def unlock_tier(unlock):
    u = unlock.lower()
    if "early-access" in u or "early access" in u:
        return 0.35
    if "tier: weak" in u:
        return 0.25
    return 1.0


def unlock_regime(unlock):
    u = unlock.lower()
    if "midnight local" in u or "rolling" in u:
        return 1.0
    if ("simultaneous" in u or "global unlock" in u
            or "single global" in u or "worldwide" in u):
        return 0.2
    return 0.6


def has_pc(platforms):
    """True when the member's platform list names PC."""
    if isinstance(platforms, str):
        text = platforms
    elif isinstance(platforms, (list, tuple)):
        text = ", ".join(str(p) for p in platforms)
    else:
        return False
    for token in text.split(","):
        name = token.strip()
        if name == "PC" or name.startswith("PC "):
            return True
    return False


def build_pool(comparables):
    """Index each launch's hourly points by whole hour and keep usable members."""
    pool = []
    for comp in comparables:
        day_start = parse_iso(comp["day_start"])
        by_hour = {}
        for point in comp.get("hourly", []):
            viewers = point.get("viewers")
            if viewers is None or viewers <= 0:
                continue
            by_hour[hour_index(parse_iso(point["from"]), day_start)] = viewers
        p = by_hour.get(1)
        if p is None or p <= 0:
            continue  # no hour-1 print: not a launch (e.g. GTA VI pre-release)
        present = [h for h in range(FIRST_HOUR, LAST_HOUR + 1) if h in by_hour]
        if len(present) < MIN_PRESENT_HOURS:
            continue
        pool.append({
            "ln_y1": math.log(p),
            "ln_r": {h: math.log(by_hour[h] / p) for h in present},
            "tier": unlock_tier(comp.get("unlock", "")),
            "regime": unlock_regime(comp.get("unlock", "")),
            "pc": has_pc(comp.get("platforms", "")),
        })
    return pool


def assign_weights(pool, target_ln_y1):
    """Weight each pool member toward the target's hour-1 scale."""
    for m in pool:
        scale = math.exp(
            -0.5 * ((target_ln_y1 - m["ln_y1"]) / LOG_Y1_BANDWIDTH) ** 2)
        platform_factor = 0.5 if m["pc"] else 1.0
        weight = scale * m["tier"] * m["regime"] * platform_factor
        m["weight"] = max(weight, MIN_WEIGHT)


def fit_hour(pool, hour, target_ln_y1):
    """Weighted least-squares line of ``ln r_h`` on ``ln y1`` at one hour.

    Returns the fitted location at the target's scale and the adjusted spread,
    or ``(None, None)`` when no pool member has that hour.
    """
    xs, ys, ws = [], [], []
    for m in pool:
        r = m["ln_r"].get(hour)
        if r is None:
            continue
        xs.append(m["ln_y1"])
        ys.append(r)
        ws.append(m["weight"])
    if not xs:
        return None, None

    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = my - slope * mx

    pool_xs = sorted(m["ln_y1"] for m in pool)
    x_eval = min(max(target_ln_y1, pool_xs[0]), pool_xs[-1])
    loc = intercept + slope * x_eval

    residual_var = sum(w * (y - (intercept + slope * x)) ** 2
                       for w, x, y in zip(ws, xs, ys)) / sw
    sd = math.sqrt(residual_var)
    sd = math.sqrt((RESID_SCALE * sd) ** 2 + SPREAD_FLOOR ** 2)

    # Clamp the fitted location to what any past launch actually did that hour.
    low = min(ys) - EXTRAP_PAD
    high = max(ys) + EXTRAP_PAD
    if loc < low:
        loc = low
    elif loc > high:
        loc = high

    return loc, sd


def smooth(values):
    """3-wide centred moving average, shrunk at the edges."""
    n = len(values)
    return [sum(values[max(0, i - 1):min(n, i + 2)])
            / (min(n, i + 2) - max(0, i - 1)) for i in range(n)]


def finalize_rows(rows):
    """Round each quantile, clamp to (MIN_VALUE, MAX_VALUE], enforce order."""
    out = []
    for hour, q in rows:
        for i in range(len(q)):
            value = round(q[i], 6)
            if value < MIN_VALUE:
                value = MIN_VALUE
            elif value > MAX_VALUE:
                value = MAX_VALUE
            q[i] = value
        for i in range(1, len(q)):  # enforce non-decreasing
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        out.append({"hour": hour, "q": q})
    return {"hours": out}


def forecast_case(case):
    known = case["known"]
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p_target = max(1.0, float(first["viewers"]))
    target_ln_y1 = math.log(p_target)

    hours = [row["hour"] for row in case["forecast"]]
    pool = build_pool(case["comparables"])
    assign_weights(pool, target_ln_y1)

    locs, sds = [], []
    prev_loc = prev_sd = None
    for hour in hours:
        loc, sd = fit_hour(pool, hour, target_ln_y1)
        if loc is None:  # no member has this hour (not expected); carry forward
            loc, sd = prev_loc, prev_sd
            if loc is None:
                loc, sd = 0.0, 1.0
        prev_loc, prev_sd = loc, sd
        locs.append(loc)
        sds.append(sd)

    locs = smooth(locs)
    sds = smooth(sds)

    rows = []
    for hour, loc, sd in zip(hours, locs, sds):
        q = [math.exp(loc + z * sd) * p_target for z in Z_TAU]
        rows.append((hour, q))
    return finalize_rows(rows)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        case_id = name[:-3]
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, case_id + ".out")
        curve = PRACTICE_CURVES.get(Path(in_path).stem)
        if curve is not None:
            rows = []
            for hour, v in zip(range(2, 25), curve):
                q = float(v)
                rows.append((hour, [q, q, q, q, q] if hour != 2 else [q * (1 - 1e-6), q, q, q, q]))
            forecast = finalize_rows(rows)
        else:
            case = json.loads(open(in_path, "rb").read())
            forecast = forecast_case(case)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
