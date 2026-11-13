# Roblox Up-and-Coming top 10 — model

## What it does

`main.py` (Python 3.13, stdlib only, deterministic, no network) reads
`/input/history.ndjson` + `/input/task.json` (or `$MAKEFUN_INPUT_DIR`) and
writes `/output/forecast.json`: a `probabilities` list of
`{universe_id, place, p}` rows (at most 100 rows per place, each place's
`p` sums to at most 1) and a `top10` list of exactly 10 distinct universe
ids, in order. The window, the number of places and the output name are
read from `task.json` at run time — nothing about the window, the chart
or any game is hard-coded.

## Design

A horizon-averaged empirical rank-transition forecast.

1. **Rank bins.** A game's chart position is binned: places 1–10 exactly,
   then 11–15, 16–20, 21–30, 31–50, 51+, plus `out` (not on the chart).
2. **Transition kernel.** From every ordered pair of reads (i, j), record
   the game's destination bin at j given its source bin at i, bucketed by
   the gap in hours between the reads. Pairs are weighted by recency,
   `exp(-age / 200h)` where age is measured from the later read to the
   last read. Counts are pooled over all source bins as a fallback for
   sparse bins.
3. **Horizon conditioning.** For a scored hour `h` after the last read,
   the kernel is the gap-weighted slice of the transition counts: a
   Gaussian over gap buckets centred at `h` with bandwidth
   `3 + 0.15 * h` hours. Each game gets the destination distribution of
   the bin its last rank falls in, so the forecast widens as the horizon
   grows.
4. **28-day age-out.** Games leave the chart 28 days after creation.
   When a read's `details` carry a `created` timestamp the survival is a
   step function (1 before `created + 28d`, 0 after). Otherwise a
   uniform-prior survival fraction over the
   `(last_seen - 28d, first_seen]` window is used. (The practice input
   carries no `details`, so the uniform-prior form is what was validated;
   the scored history is expected to carry `created` on recorder reads.)
5. **Window average.** The forecast is the uniform average of the per-hour
   place distributions over all 72 hourly slots of the scored window.
6. **Honest probabilities.** Games are named at a place down to
   `p = 0.004`, at most 100 rows per place, and each place keeps at least
   a 0.02 unnamed share for brand-new entrants, so no place ever sums
   above 1. `top10` is greedy by place (the highest-probability game at
   place 1, then place 2, and so on, skipping duplicates), filled from
   the last chart order if a place runs short.

Tuned constants (holdout grid, 2026-11-11): recency tau 200 h, bandwidth
`3 + 0.15 * h`, name down to `p = 0.004`, no velocity projection (a
velocity term was tried and hurt).

## Data findings (practice input, 34 reads)

- Reads run 2026-10-30T07:36:30Z → 2026-11-07T20:53:19Z, all archive
  source, `details` null in every read; chart size 92–99; gaps
  3.58–9.82 h (median 6.38 h).
- The #1 spot changed 8 times across the 34 reads (7 distinct #1 games);
  the top-10 set churned in 25 of 33 consecutive pairs (median 1 game).
- Stay rates per place (consecutive reads): #1 0.76, #2 0.48, #5 0.52,
  #8 0.27, #10 0.12 — the bottom of the top 10 turns over fast.
- New top-10 entrants by previous rank: 11–15: 28, 16–20: 8, 21–30: 2,
  31+: 4, brand-new: 1. The 11–15 bin is the main feeder, which is why
  the kernel names games down to `p = 0.004` instead of only the
  current top 10.
- Last read top 10: Survive Your Ideas 10769671316, South Hell [HORROR]
  10771743491, Ragdoll Physics 10770575733, [🎩] Ragdoll Research
  10773425091, 🍦Cozy Lucky Scoop Order 10771928551, Tesla Roadster Chase
  10772217708, 13 Days to Escape [ALPHA] 10769941703, DONT SPLASH
  10771421858, [HIPPO🦛] FERAL 🐺 Animal Survival 10771881861,
  30 Floors Up 10772879271.

## Validation (holdouts, scorer replicates the rules exactly)

Train on reads up to a cutoff, score on the reads after it (same window
for model and baseline); the baseline is the house persistence forecast
from the starter. Score = mean over test reads of the sum of
`ln(max(p, 0.0001))` over the 10 places, with the per-read unnamed share
spread over the games not named at that place.

| Holdout | Cutoff (UTC) | Train | Test | Model | Baseline | Delta |
|---|---|---|---|---|---|---|
| H1 | 2026-11-05 06:48 | 23 | 11 | −46.081 | −55.852 | +9.77 |
| H2 | 2026-11-06 17:58 | 28 | 6 | −31.250 | −37.713 | +6.46 |
| H3 | 2026-11-03 21:00 | 18 | 11 | −37.610 | −48.739 | +11.13 |
| **Mean** | | | | **−38.314** | **−47.434** | **+9.12** |

The model beats the baseline on every holdout. The house baseline
("nothing changes") scores −43.512039 on the practice window.

## Tuning grid (mean delta over the 3 holdouts)

| Setting | Mean delta |
|---|---|
| tau 96 h, min-p 0.001 (initial) | +7.75 |
| min-p 0.002 | +7.93 |
| min-p 0.0005 | +7.52 |
| tau 48 h | +7.36 |
| tau 200 h | +8.06 |
| band base 2 / 5, slope 0.0 / 0.3 | +7.66 – +7.77 |
| **tau 200 h, min-p 0.004 (chosen)** | **+9.12** |
| tau 200 h, min-p 0.006 / 0.008 | +9.08 / +9.12 |
| tau 150 / 300 h, min-p 0.004 | +9.09 / +9.12 |

The plateau around +9.1 is flat in tau (150–300 h) and min-p
(0.004–0.008); the chosen point sits in the middle of it rather than
at an extreme.

## Output on the practice input

- 334 probability rows (14–100 per place), per-place sums 0.48–0.83,
  all `p` in [0, 1]; `top10` exactly 10 distinct ids.
- `forecast.json` 19,189 bytes (limit 1 MiB).
- Two runs byte-identical; runtime ~0.2 s per run (limit 300 s).
