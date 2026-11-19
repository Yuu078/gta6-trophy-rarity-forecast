// GTA VI trophy rarity forecaster — make.fun gta6-trophy-rarity-forecast-01.
//
// Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
// quantiles (5/25/50/75/95%) of its Exophase earn rate at the case target,
// scored by pinball loss on the logit scale.
//
// Method (per trophy, all from the case file alone):
//   1. Collapse the history into distinct observations. Exophase refreshes a
//      game's counts on its own schedule, so consecutive reads that print
//      identical rates carry no new information — they are the same stale
//      page. Only reads whose rate vector actually changed count, dated at
//      the first read that showed them.
//   2. Fit a least-squares line of logit(rate) against ln(1 + hours since
//      the release day began, 2026-11-19T00:00:00Z) over those distinct
//      observations. Log-time is the right clock: earn rates move fast in
//      the first hours as the launch cohort plays through, then flatten as
//      the cohort matures, while fresh players keep joining and diluting
//      the rates — the fitted slope is the net of both forces, read off the
//      data instead of assumed.
//   3. Extrapolate the line to the target. The slope is shrunk toward zero
//      (factor n / (n + 2)), the part of the move past the last observed
//      time counts half, and the total move from the last read is capped
//      at 2.5 logit units, so a short or noisy history cannot fling the
//      median somewhere absurd.
//   4. With a single distinct observation there is no trend to fit. Then the
//      honest forecast is a dilution prior: the next Exophase refresh folds
//      in a wave of new players who have earned far fewer trophies, so the
//      median sits below the last rate and the spread is wide enough to
//      cover the refresh landing either side of it.
//   5. Spread: a random-walk term (0.5 logit per sqrt day, as the house
//      baseline), the fit's residual and leverage, and a refresh term for
//      refreshes the case has not seen yet — combined in quadrature, floored
//      and capped so the 90% band stays honest without exploding.
//
// Node 22, standard library only. Deterministic: no clock, no randomness,
// inputs read in name order, outputs rounded to 4 decimals.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels the contest scores, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at those levels. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release day began here; the log-time clock starts at this instant. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Random-walk spread, logit units per square-root day. */
const WALK_SIGMA = 0.5;
/** Extra spread for refreshes not yet seen, logit units. */
const REFRESH_SIGMA = 0.35;
/** Median shift (logit) when the history holds a single distinct read. */
const FLAT_SHIFT = -0.7;
/** Spread (logit) when the history holds a single distinct read. */
const FLAT_SIGMA = 1.3;

interface Trophy {
  id: number;
  tier: string;
}

interface Read {
  at: string;
  rates: Array<number | null>;
}

interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours from the release-day start to an ISO instant, floored at 0. */
const hoursSinceRelease = (iso: string): number =>
  Math.max(0, (Date.parse(iso) - RELEASE_MS) / 3600000);
/** The log-time coordinate of a read instant. */
const timeCoord = (iso: string): number => Math.log1p(hoursSinceRelease(iso));

/** Tier prior, used only when a trophy has no usable rate at all. */
const tierPrior = (tier: string): number => {
  if (tier === 'platinum') return 0.5;
  if (tier === 'gold') return 2;
  if (tier === 'silver') return 5;
  return 10;
};

/**
 * Forecast one trophy: [median logit at target, spread in logit units].
 * `obs` holds the distinct (timeCoord, logitRate) observations, oldest first.
 */
function forecastOne(
  obs: Array<{ x: number; y: number }>,
  lastRate: number,
  targetX: number,
  horizonDays: number,
): [number, number] {
  const lastLogit = logit(lastRate);
  if (obs.length === 0) return [logit(tierPrior('bronze')), 1.5];
  if (obs.length === 1) {
    // No trend exists yet: expect the next refresh to dilute the rate.
    // A trophy sitting at the 100% ceiling has the most dilution exposure
    // in percentage points — a fresh cohort cannot all hold it at once —
    // so its band is widened further on the logit scale.
    const flatSigma = lastRate >= 99.9 ? 2.0 : FLAT_SIGMA;
    const sigma = Math.sqrt(
      flatSigma * flatSigma + WALK_SIGMA * WALK_SIGMA * horizonDays,
    );
    return [lastLogit + FLAT_SHIFT, Math.min(2.2, Math.max(0.4, sigma))];
  }

  // Least squares of y on x over the distinct observations.
  const n = obs.length;
  let sx = 0;
  let sy = 0;
  for (const o of obs) {
    sx += o.x;
    sy += o.y;
  }
  const xBar = sx / n;
  const yBar = sy / n;
  let sxx = 0;
  let sxy = 0;
  for (const o of obs) {
    const dx = o.x - xBar;
    sxx += dx * dx;
    sxy += dx * (o.y - yBar);
  }
  // Slope shrunk toward zero: few distinct reads earn little extrapolation.
  const rawSlope = sxx > 1e-12 ? sxy / sxx : 0;
  const slope = rawSlope * (n / (n + 2));
  // The unshrunk line, used only to measure the fit's residual scatter.
  const intercept = yBar - rawSlope * xBar;
  // Anchor the forecast at the LAST observed rate and move from there by
  // the fitted slope: the line's level can sit off the latest read when
  // the path curves, but the latest read is the freshest truth. The move
  // past the last observed time counts half — a trend fitted inside the
  // window is not trusted at full strength beyond it.
  const lastX = obs[obs.length - 1].x;
  const lastY = obs[obs.length - 1].y;
  const beyond = Math.max(0, targetX - lastX);
  let median =
    lastY +
    slope * (Math.min(targetX, lastX) - lastX) +
    slope * 0.5 * beyond;
  // Cap the move from the last read: trends do not teleport rates.
  median = Math.min(lastLogit + 2.5, Math.max(lastLogit - 2.5, median));

  // Residual scatter of the fit.
  let sse = 0;
  for (const o of obs) {
    const r = o.y - (intercept + (sxx > 1e-12 ? sxy / sxx : 0) * o.x);
    sse += r * r;
  }
  const resid = n > 2 ? Math.sqrt(sse / (n - 2)) : 0.12;
  const residCapped = Math.min(1.2, Math.max(0.08, resid));
  const leverage =
    1 + 1 / n + (sxx > 1e-12 ? (targetX - xBar) * (targetX - xBar) / sxx : 1);
  const fitSigma = residCapped * Math.sqrt(leverage);
  const walk = WALK_SIGMA * Math.sqrt(horizonDays);
  const sigma = Math.sqrt(
    walk * walk + fitSigma * fitSigma + REFRESH_SIGMA * REFRESH_SIGMA,
  );
  return [median, Math.min(2.2, Math.max(0.4, sigma))];
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const horizonDays = c.horizon_minutes / 1440;
  const targetX = timeCoord(c.target);

  // Collapse the history: keep a read only when its rate vector changed.
  const distinct: Read[] = [];
  let prevKey = '';
  for (const h of c.history) {
    const key = JSON.stringify(h.rates);
    if (key !== prevKey) {
      distinct.push(h);
      prevKey = key;
    }
  }

  const trophies = c.trophies.map((t, i) => {
    const obs: Array<{ x: number; y: number }> = [];
    let lastRate: number | null = null;
    for (const h of distinct) {
      const r = h.rates[i];
      if (r === null || r === undefined || !Number.isFinite(r)) continue;
      obs.push({ x: timeCoord(h.at), y: logit(r) });
      lastRate = r;
    }
    if (lastRate === null) {
      // No usable rate: fall back to the tier prior, wide spread.
      const m = logit(tierPrior(t.tier));
      const q = Z.map((z) =>
        round4(clampRate(invLogit(m + z * 1.5))),
      );
      return { id: t.id, q: q.sort((a, b) => a - b) };
    }
    const [median, sigma] = forecastOne(obs, lastRate, targetX, horizonDays);
    const q = Z.map((z) => round4(clampRate(invLogit(median + z * sigma))));
    // Enforce non-decreasing after rounding.
    for (let j = 1; j < q.length; j++) {
      if (q[j] < q[j - 1]) q[j] = q[j - 1];
    }
    return { id: t.id, q };
  });
  writeFileSync(
    join(OUTPUT_DIR, `${id}.out`),
    `${JSON.stringify({ trophies })}\n`,
  );
}
