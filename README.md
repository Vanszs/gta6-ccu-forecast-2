[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Vanszs/gta6-ccu-forecast-2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Vanszs/gta6-ccu-forecast-2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

A deterministic forecaster for make.fun's `gta6-twitch-ccu-forecast-01` bounty: it predicts Grand Theft Auto VI's concurrent Twitch viewers for each of release-day hours 2..24 as five quantiles. The model works in log ratios to hour 1's print — for every comparable launch it fits a per-hour weighted least-squares line of `ln(viewers_h / viewers_1)` on `ln(viewers_1)` (bigger launches have flatter days), weights the pool by scale, unlock tier, unlock regime and platform, adds a scaled residual spread, clamps the fitted location to the range any past launch actually hit that hour, and smooths the curve across neighbouring hours before exponentiating the quantiles back to viewer counts. Standard library only, no network, deterministic byte-for-byte.

## Run it

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=/tmp/opencode/kit-gta6/input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```
