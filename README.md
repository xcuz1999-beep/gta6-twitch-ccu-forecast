# GTA VI launch-day Twitch CCU quantile forecast

Entry for make.fun competition `gta6-twitch-ccu-forecast-01` ($250, 20 places):
forecast, for each of hours 2–24 of GTA VI's release day (2026-11-19 UTC),
five quantiles (5%, 25%, 50%, 75%, 95%) of the mean concurrent viewers in
Twitch's "Grand Theft Auto VI" category, given only the first hour's count.

## Model

Scale-conditioned comparable-launch quantile forecaster (`main.py`, Python
3.13, stdlib only, no network, deterministic). At run time it reads the
case's own `comparables` — past big game launches' hourly Twitch category
averages — keeps the launches that are fair analogues for GTA VI (a console
game unlocking at midnight local time, rolling country by country, no paid
early access), and fits per forecast hour a linear regression of the log
viewer ratio `log(viewers_h / viewers_1)` on the log size of hour 1. The
slope is negative at every later hour: launches whose first hour is already
huge grow little afterwards, while launches whose first hour only covers the
earliest timezones roughly double or triple by the US evening peak. The
median is a 75/25 blend of that regression with the plain median log ratio
of the same fit set; the spread is the regression's residual SD of log
ratios — an honest spread that widens over the day. Fitted coefficients are
also hardcoded in `main.py` as a fallback if a case file carries too few
usable comparables.

## Validation

Leave-one-out over the 14 fit launches (mean pinball loss on the log scale,
the contest's scorer): **0.229**, versus **0.363** for the house persistence
baseline on the same launches (house baseline's published practice loss:
0.387). See `gta6-work/analyze*.py` in the build notes for the full
backtest; the sealed practice grade is reported in the submission record.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
