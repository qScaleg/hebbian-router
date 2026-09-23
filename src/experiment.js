// Runs N seeds x T tasks through each router, measures cumulative success
// rate, regret vs. an oracle that always knows the true best agent, and
// adaptation speed around a mid-run skill drift. Prints a table and writes
// results/results.json.
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createWorld, AGENTS, TASK_TYPES } from './simulator.js';
import { KeywordRouter, ThompsonRouter, HebbianRouter, PheromoneRouter } from './routers.js';
import { BioMeshRouter } from './biomesh.js';
import { mulberry32 } from './prng.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// eta/epsilon/temperature were picked by a small grid search (see
// README "Hyperparameter selection") to give the reward-modulated Hebbian
// router a fair shot rather than leaving it at arbitrary defaults.
const HEBBIAN_OPTS = { eta: 0.5, epsilon: 0.02, temperature: 0.05 };

// ruflo pheromone-adaptive (ADR-330, see src/pheromone.js). 'pheromone' and
// 'pheromone-select' run ruflo's DEFAULT_APSC_CONFIG unchanged (live mode).
// The '-tuned' rows use the best config found by src/grid-pheromone.js with
// the same grid budget the Hebbian router got (<=60 configs x 8 seeds x 2000).
// Gate: every top config ties at the keyword router's score (none of them
// ever suspends an agent); this is the first in grid order.
export const PHEROMONE_GATE_TUNED = { emaDecay: 0.5, pruningFactor: 0.4, explorationRate: 0.05, roleMode: 'per-agent' };
// Select: pruningFactor/explorationRate did not change the score; roleMode
// 'shared' and emaDecay 0.7 did.
export const PHEROMONE_SELECT_TUNED = { emaDecay: 0.7, pruningFactor: 0.4, explorationRate: 0.05, roleMode: 'shared' };

// neuralmesh-biological mechanisms ported onto routing (src/biomesh.js).
// Each variant uses the best config from src/grid-biomesh.js (tuned
// separately, <=36 configs x 8 seeds x 2000). 'biomesh-full-defaults' runs
// the library's own defaults (lam 10, lr 0.1, consolidation every 40).
export const BIOMESH_TUNED = {
  'biomesh-sgd': { consolidation: false, replay: false, neurogenesis: false, lr: 0.1, temperature: 0.1 },
  'biomesh-consolidation-only': { replay: false, neurogenesis: false, lr: 0.1, temperature: 0.1, lam: 1, consolidateEvery: 200 },
  'biomesh-consolidation+replay': { neurogenesis: false, lr: 0.3, temperature: 0.1, lam: 1, consolidateEvery: 200 },
  'biomesh-full': { lr: 0.1, temperature: 0.1, lam: 1, consolidateEvery: 200 },
  'biomesh-full-defaults': {},
};

export function buildRouters(seed) {
  return [
    new KeywordRouter(AGENTS),
    new ThompsonRouter(AGENTS, TASK_TYPES, mulberry32(seed * 7919 + 1)),
    new HebbianRouter(AGENTS, { rng: mulberry32(seed * 7919 + 2), decay: 0.01, ...HEBBIAN_OPTS }),
    new HebbianRouter(AGENTS, { rng: mulberry32(seed * 7919 + 3), meshStyle: true, ...HEBBIAN_OPTS }),
    new PheromoneRouter(AGENTS, { rng: mulberry32(seed * 7919 + 4), mode: 'gate' }),
    new PheromoneRouter(AGENTS, { rng: mulberry32(seed * 7919 + 5), mode: 'select' }),
    new PheromoneRouter(AGENTS, {
      rng: mulberry32(seed * 7919 + 6), mode: 'gate', name: 'pheromone-tuned', ...PHEROMONE_GATE_TUNED,
    }),
    new PheromoneRouter(AGENTS, {
      rng: mulberry32(seed * 7919 + 7), mode: 'select', name: 'pheromone-select-tuned', ...PHEROMONE_SELECT_TUNED,
    }),
    ...Object.entries(BIOMESH_TUNED).map(([name, cfg], i) => new BioMeshRouter(AGENTS, TASK_TYPES, {
      rng: mulberry32(seed * 7919 + 20 + i), name, ...cfg,
    })),
  ];
}

/** Run one seed's worth of tasks through every router. Uses common random
 * numbers (same task stream and same outcome draw per task index) so
 * routers are compared on equal footing rather than lucky/unlucky draws.
 * Routers each own their RNG, so adding a router never changes another's
 * results. `routers` can be overridden (the grid search passes one). */
export function runOne(seed, T, driftPoint, routers = buildRouters(seed)) {
  const world = createWorld({ driftPoint });
  const taskRng = mulberry32(seed * 104729 + 3);
  const outcomeRng = mulberry32(seed * 613 + 17);

  const series = {};
  for (const r of routers) series[r.name] = { rewards: [], regrets: [] };

  for (let i = 0; i < T; i++) {
    const task = world.sampleTask(taskRng, i);
    const oracleAgent = world.oracleAgent(task.type, i);
    const oracleProb = world.successProbability(task.type, oracleAgent, i);
    const u = outcomeRng(); // shared draw across routers for this task

    for (const r of routers) {
      const agent = r.select(task);
      const p = world.successProbability(task.type, agent, i);
      const reward = u < p ? 1 : 0;
      r.update(task, agent, reward);
      series[r.name].rewards.push(reward);
      series[r.name].regrets.push(oracleProb - p);
    }
  }

  return { series, T, driftPoint };
}

function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function sum(arr) {
  return arr.reduce((a, b) => a + b, 0);
}

/** Summarize one seed's run into per-router scalar metrics. */
export function summarize({ series, T, driftPoint }) {
  const windowSize = Math.max(1, Math.min(50, T - driftPoint));
  const out = {};
  for (const [name, { rewards, regrets }] of Object.entries(series)) {
    const preDrift = rewards.slice(0, driftPoint);
    const earlyPostDrift = rewards.slice(driftPoint, driftPoint + windowSize);
    const latePostDrift = rewards.slice(Math.max(driftPoint, T - windowSize), T);
    out[name] = {
      successRate: mean(rewards),
      totalRegret: sum(regrets),
      preDriftSuccessRate: mean(preDrift),
      earlyPostDriftSuccessRate: mean(earlyPostDrift),
      latePostDriftSuccessRate: mean(latePostDrift),
    };
  }
  return out;
}

function avgAcross(perSeedSummaries, routerNames) {
  const out = {};
  for (const name of routerNames) {
    const rows = perSeedSummaries.map((s) => s[name]);
    out[name] = {
      successRate: mean(rows.map((r) => r.successRate)),
      totalRegret: mean(rows.map((r) => r.totalRegret)),
      preDriftSuccessRate: mean(rows.map((r) => r.preDriftSuccessRate)),
      earlyPostDriftSuccessRate: mean(rows.map((r) => r.earlyPostDriftSuccessRate)),
      latePostDriftSuccessRate: mean(rows.map((r) => r.latePostDriftSuccessRate)),
    };
  }
  return out;
}

export function runExperiment({ seeds = 30, T = 2000 } = {}) {
  const driftPoint = Math.floor(T / 2);
  const routerNames = buildRouters(0).map((r) => r.name);
  const perSeed = [];
  for (let seed = 1; seed <= seeds; seed++) {
    perSeed.push(summarize(runOne(seed, T, driftPoint)));
  }
  const aggregate = avgAcross(perSeed, routerNames);
  const paired = pairedComparisons(perSeed, 'hebbian', routerNames.filter((n) => n !== 'hebbian'));
  // Stability-plasticity: each biomesh mechanism vs plain SGD on the same
  // table, per drift window. Here + means the variant beat SGD.
  const bio = routerNames.filter((n) => n.startsWith('biomesh-') && n !== 'biomesh-sgd');
  const bioVsSgd = {};
  if (routerNames.includes('biomesh-sgd')) {
    for (const name of bio) {
      bioVsSgd[name] = Object.fromEntries(DRIFT_METRICS.map((m) => [m, pairedStat(
        perSeed.map((s) => s[name][m] - s['biomesh-sgd'][m]),
      )]));
    }
  }
  // biomesh-full against each learning baseline (biomesh-full minus other).
  const bioFull = routerNames.includes('biomesh-full')
    ? Object.fromEntries(['hebbian', 'thompson'].map((other) => [other, Object.fromEntries(
      DRIFT_METRICS.map((m) => [m, pairedStat(perSeed.map((s) => s['biomesh-full'][m] - s[other][m]))]),
    )]))
    : {};
  return { seeds, T, driftPoint, routerNames, aggregate, paired, bioVsSgd, bioFull };
}

// Two-sided 95% Student-t critical values; df beyond the table uses 1.96.
const T95 = { 1: 12.706, 2: 4.303, 4: 2.776, 7: 2.365, 9: 2.262, 19: 2.093, 29: 2.045 };

/** Paired per-seed differences (ref - other) on successRate and totalRegret:
 * mean, 95% CI half-width, and how many seeds each side won. The routers
 * share task and outcome streams per seed, so pairing removes most noise. */
const DRIFT_METRICS = ['successRate', 'preDriftSuccessRate', 'earlyPostDriftSuccessRate', 'latePostDriftSuccessRate'];

/** Mean, 95% t-CI half-width and win counts of a list of paired differences. */
export function pairedStat(diffs) {
  const n = diffs.length;
  const t = T95[n - 1] ?? 1.96;
  const m = mean(diffs);
  const sd = n > 1 ? Math.sqrt(sum(diffs.map((d) => (d - m) ** 2)) / (n - 1)) : 0;
  return {
    mean: m,
    ci95: n ? t * sd / Math.sqrt(n) : 0,
    refWins: diffs.filter((d) => d > 1e-12).length,
    otherWins: diffs.filter((d) => d < -1e-12).length,
  };
}

export function pairedComparisons(perSeed, ref, others) {
  const out = {};
  const stat = pairedStat;
  for (const other of others) {
    out[other] = {
      successRate: stat(perSeed.map((s) => s[ref].successRate - s[other].successRate)),
      // Lower regret is better, so ref "wins" when other - ref > 0.
      totalRegret: stat(perSeed.map((s) => s[other].totalRegret - s[ref].totalRegret)),
    };
  }
  return out;
}

function fmt(x) {
  return x.toFixed(4);
}

function printTable({ seeds, T, driftPoint, routerNames, aggregate, paired, bioVsSgd, bioFull }) {
  console.log(`\nHebbian router experiment: ${seeds} seeds x ${T} tasks (drift at task ${driftPoint})\n`);
  const cols = ['router', 'successRate', 'totalRegret', 'preDrift', 'earlyPostDrift', 'latePostDrift'];
  const rows = routerNames.map((name) => {
    const a = aggregate[name];
    return [
      name,
      fmt(a.successRate),
      fmt(a.totalRegret),
      fmt(a.preDriftSuccessRate),
      fmt(a.earlyPostDriftSuccessRate),
      fmt(a.latePostDriftSuccessRate),
    ];
  });
  const widths = cols.map((c, i) => Math.max(c.length, ...rows.map((r) => r[i].length)));
  const line = (r) => r.map((c, i) => c.padEnd(widths[i])).join('  ');
  console.log(line(cols));
  console.log(widths.map((w) => '-'.repeat(w)).join('  '));
  for (const r of rows) console.log(line(r));
  console.log('\nPaired vs hebbian over seeds (hebbian minus other; regret sign flipped so + favours hebbian):');
  for (const [name, p] of Object.entries(paired)) {
    const f = (x) => `${x.mean >= 0 ? '+' : ''}${x.mean.toFixed(4)} +/- ${x.ci95.toFixed(4)} (wins ${x.refWins}-${x.otherWins})`;
    console.log(`  ${name.padEnd(24)} success ${f(p.successRate)}   regret ${f(p.totalRegret)}`);
  }
  console.log('\nPaired vs biomesh-sgd (variant minus sgd; + = variant better), all / pre / earlyPost / latePost:');
  for (const [name, p] of Object.entries(bioVsSgd)) {
    const f = (x) => `${x.mean >= 0 ? '+' : ''}${x.mean.toFixed(3)}+/-${x.ci95.toFixed(3)}`;
    console.log(`  ${name.padEnd(22)} ${DRIFT_METRICS.map((m) => f(p[m])).join('  ')}`);
  }
  console.log('\nPaired biomesh-full minus other (+ = biomesh-full better; seed wins full-other), all / pre / earlyPost / latePost:');
  for (const [name, p] of Object.entries(bioFull)) {
    const f = (x) => `${x.mean >= 0 ? '+' : ''}${x.mean.toFixed(3)}+/-${x.ci95.toFixed(3)} (${x.refWins}-${x.otherWins})`;
    console.log(`  vs ${name.padEnd(10)} ${DRIFT_METRICS.map((m) => f(p[m])).join('  ')}`);
  }
  console.log('');
}

function writeResults(result) {
  const outDir = join(__dirname, '..', 'results');
  mkdirSync(outDir, { recursive: true });
  const outFile = join(outDir, 'results.json');
  writeFileSync(outFile, JSON.stringify(result, null, 2));
  console.log(`Wrote ${outFile}`);
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const result = runExperiment({ seeds: 30, T: 2000 });
  printTable(result);
  writeResults(result);
}
