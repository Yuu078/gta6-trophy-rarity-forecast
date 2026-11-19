# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yuu078/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yuu078/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster that predicts each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate one week after launch, as five quantiles per trophy, from the house's recorded reads of the trophy page.

## What it does

`main.ts` (Node 22, standard library only) reads every case file in the input directory and writes one forecast per case: for each of the game's 55 trophies, the 5%, 25%, 50%, 75% and 95% quantiles of its earn rate at the case target, scored by pinball loss on the logit scale.

How it forecasts, per trophy:

- **Stale reads are collapsed first.** Exophase refreshes a game's counts on its own schedule, so consecutive reads printing identical rates are the same stale page. Only reads whose rate vector actually changed count as observations, dated at the first read that showed them.
- **A trend is fitted on the logit scale.** Least squares of logit(rate) against ln(1 + hours since the release day began), over those distinct observations. Log-time is the right clock: rates move fast in the first hours as the launch cohort plays through, then flatten as the cohort matures, while fresh players keep joining and diluting the rates — the fitted slope is the net of both forces, read off the data.
- **The extrapolation is anchored and damped.** The forecast starts from the last observed rate and moves by the fitted slope, shrunk toward zero (factor n / (n + 2)); the move past the last observed time counts half, and the total move is capped at 2.5 logit units.
- **A flat history gets a dilution prior.** With a single distinct observation there is no trend to fit, so the median sits below the last rate (the next refresh folds in new players who have earned far fewer trophies) and the spread is wide.
- **The spread is honest.** A random-walk term (0.5 logit per square-root day), the fit's residual and leverage, and a term for refreshes not yet seen, combined in quadrature — so the 90% band widens with the horizon and with how little the history actually says.

The program is deterministic (no clock, no randomness), runs in well under the 8-second cap per case, and uses no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

It reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `/output/<id>.out` (else `$MAKEFUN_OUTPUT_DIR`) as `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, quantiles non-decreasing, each 0 to 100.

---

<sub>Entered on [make.fun](https://make.fun/?ref=pd4x8eme&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
