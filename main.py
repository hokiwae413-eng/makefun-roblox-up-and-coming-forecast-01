#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 for the scored window.

Horizon-averaged empirical rank-transition forecast. Reads the chart
history from /input (or $MAKEFUN_INPUT_DIR) and writes
/output/forecast.json (or $MAKEFUN_OUTPUT_DIR).

Model:
  1. Rank-transition kernel: from every ordered pair of reads, the
     distribution of a game's rank bin at the later read given its rank
     bin at the earlier read, bucketed by the gap in hours and weighted
     by recency (exp(-age/96h)), so recent dynamics dominate.
  2. Horizon conditioning: for a scored hour h after the last read, a
     Gaussian slice of those counts centred at h (bandwidth 3 + 0.15*h).
  3. Age-out: games leave the chart 28 days after creation. Exact from
     details.created when present; otherwise a uniform-prior survival
     fraction over the (last_seen - 28d, first_seen] window.
  4. The forecast is the uniform average over the window's hourly slots.
  5. Honest probabilities: named down to p = 0.001 (at most 100
     rows/place), each place keeps at least 0.02 unnamed for brand-new
     entrants, no place sums above 1. top10 is greedy by place.

Standard library only, deterministic, no network, no keys, no model
calls. Two runs on the same history write byte-identical output.
"""
import json
import math
import os
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Tuned constants (holdout grid, 2026-11-11): tau 200 h and min-p 0.004
# beat tau 96 / min-p 0.001 by ~1.4 nats (mean delta +9.12 vs +7.75).
RECENCY_TAU_H = 200.0
BAND_BASE_H = 3.0
BAND_SLOPE = 0.15
NAME_MIN_P = 0.004
MAX_ROWS_PER_PLACE = 100
MIN_UNNAMED = 0.02
AGE_OUT_DAYS = 28.0


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def rank_bin(rank):
    """Rank (1-based, or None if not on the chart) -> bin 1..16.

    Bins 1-10 are exact places; 11 = 11-15, 12 = 16-20, 13 = 21-30,
    14 = 31-50, 15 = 51+, 16 = out (not on the chart).
    """
    if rank is None:
        return 16
    if rank <= 10:
        return rank
    if rank <= 15:
        return 11
    if rank <= 20:
        return 12
    if rank <= 30:
        return 13
    if rank <= 50:
        return 14
    return 15


def load_history(path):
    reads = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    return reads


def slice_dist(bucket_counts, h, sigma):
    """Gaussian-weighted destination distribution from {gap: {dst: count}}.

    Returns {dst_bin: probability} or None when there is no weight.
    """
    num = {}
    den = 0.0
    inv = 1.0 / (2.0 * sigma * sigma)
    for gap, dests in bucket_counts.items():
        w = math.exp(-((gap - h) ** 2) * inv)
        if w < 1e-12:
            continue
        for db, c in dests.items():
            num[db] = num.get(db, 0.0) + w * c
            den += w * c
    if den <= 0.0:
        return None
    return {db: v / den for db, v in num.items()}


def survival(uid, slot_time, created, first_seen, last_seen):
    """Probability the game is still on the chart at slot_time."""
    c = created.get(uid)
    if c is not None:
        return 1.0 if slot_time < c + timedelta(days=AGE_OUT_DAYS) else 0.0
    fs = first_seen.get(uid)
    ls = last_seen.get(uid)
    if fs is None or ls is None:
        return 1.0
    span = timedelta(days=AGE_OUT_DAYS).total_seconds()
    denom = (fs - ls).total_seconds() + span
    if denom <= 0.0:
        return 1.0
    numer = (fs - slot_time).total_seconds() + span
    s = numer / denom
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    window = task.get("window", {})
    w_from = parse_time(window["from"])
    slots = int(window.get("slots", 72))
    slot_min = int(window.get("slot_minutes", 60))
    out_name = task.get("output", "forecast.json")

    reads = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    if not reads:
        out = {"probabilities": [], "top10": []}
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
            f.write(json.dumps(out, sort_keys=True) + "\n")
        return

    read_times = []
    bin_maps = []
    first_seen = {}
    last_seen = {}
    created = {}
    for r in reads:
        t = parse_time(r["at"])
        read_times.append(t)
        bm = {}
        for g in r.get("up_and_coming") or []:
            uid = g["universe_id"]
            bm[uid] = rank_bin(g["rank"])
            if uid not in first_seen:
                first_seen[uid] = t
            last_seen[uid] = t
        bin_maps.append(bm)
        det = r.get("details")
        if det:
            for key, d in det.items():
                if not isinstance(d, dict):
                    continue
                try:
                    uid = int(key)
                except (TypeError, ValueError):
                    continue
                c = d.get("created")
                if c and uid not in created:
                    created[uid] = parse_time(c)

    t_last = read_times[-1]

    # Transition counts: counts[src_bin][gap_bucket][dst_bin], recency-weighted.
    counts = {}
    global_counts = {}
    for j in range(len(reads)):
        tj = read_times[j]
        age_h = (t_last - tj).total_seconds() / 3600.0
        w_rec = math.exp(-age_h / RECENCY_TAU_H)
        mj = bin_maps[j]
        for i in range(j):
            gap = int(round((tj - read_times[i]).total_seconds() / 3600.0))
            if gap < 1:
                gap = 1
            mi = bin_maps[i]
            for uid in set(mi) | set(mj):
                sb = mi.get(uid, 16)
                db = mj.get(uid, 16)
                d = counts.setdefault(sb, {}).setdefault(gap, {})
                d[db] = d.get(db, 0.0) + w_rec
                g = global_counts.setdefault(gap, {})
                g[db] = g.get(db, 0.0) + w_rec

    last_map = bin_maps[-1]
    all_uids = set()
    for bm in bin_maps:
        all_uids.update(bm.keys())
    src_bin = {uid: last_map.get(uid, 16) for uid in all_uids}

    slot_times = [w_from + timedelta(minutes=slot_min * s) for s in range(slots)]
    acc = {uid: [0.0] * (places + 1) for uid in all_uids}
    for st in slot_times:
        h = (st - t_last).total_seconds() / 3600.0
        if h < 0.0:
            h = 0.0
        sigma = BAND_BASE_H + BAND_SLOPE * h
        global_d = slice_dist(global_counts, h, sigma) or {}
        dists = {}
        for b in range(1, 17):
            d = slice_dist(counts.get(b, {}), h, sigma)
            dists[b] = d if d is not None else global_d
        for uid in all_uids:
            d = dists[src_bin[uid]]
            if not d:
                continue
            s = survival(uid, st, created, first_seen, last_seen)
            if s <= 0.0:
                continue
            row = acc[uid]
            for place in range(1, places + 1):
                p = d.get(place, 0.0)
                if p:
                    row[place] += p * s

    nslots = float(len(slot_times))
    for uid in all_uids:
        row = acc[uid]
        for place in range(1, places + 1):
            row[place] /= nslots

    probabilities = []
    by_place = {}
    for place in range(1, places + 1):
        cand = [(uid, acc[uid][place]) for uid in all_uids]
        total = sum(p for _, p in cand)
        cap = 1.0 - MIN_UNNAMED
        if total > cap and total > 0.0:
            scale = cap / total
            cand = [(uid, p * scale) for uid, p in cand]
        cand = [(uid, p) for uid, p in cand if p >= NAME_MIN_P]
        cand.sort(key=lambda x: (-x[1], x[0]))
        rows = []
        for uid, p in cand[:MAX_ROWS_PER_PLACE]:
            r = {"universe_id": uid, "place": place, "p": round(p, 6)}
            probabilities.append(r)
            rows.append((uid, r["p"]))
        by_place[place] = rows

    top10 = []
    used = set()
    for place in range(1, places + 1):
        for uid, _p in by_place.get(place, []):
            if uid not in used:
                top10.append(uid)
                used.add(uid)
                break
    if len(top10) < places:
        last_order = sorted(last_map, key=lambda u: last_map[u])
        for uid in last_order:
            if uid not in used:
                top10.append(uid)
                used.add(uid)
            if len(top10) >= places:
                break
    top10 = top10[:places]

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(out, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
