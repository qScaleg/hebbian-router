// Grid search over ruflo pheromone-adaptive's free parameters, with the same
// budget the Hebbian router got (README "Hyperparameter selection": at most
// 60 configurations x 8 seeds x 2000 tasks, drift at 1000). Seeds 1-8, the
// first 8 of the 30 evaluation seeds -- the README does not record which
// seeds the Hebbian search used, so this picks the choice that is at least
// as generous to pheromone as tuning on held-out seeds would be.
//
// Grid (48 configs per mode, both modes searched separately):
//   emaDecay        {0.5, 0.7, 0.85, 0.95}   ruflo default 0.85
//   pruningFactor   {0.4, 0.6, 0.8}          ruflo default 0.6
//   explorationRate {0.05, 0.1}              ruflo default 0.1
//   roleMode        {per-agent, shared}      hooks default per-agent
// Held at ruflo defaults: alpha/beta/gamma (latency and consensus are
// neutral constants here, so the weights only rescale the success term),
// reactivationThreshold 0.75, minSamples 3, minActiveAgents 3,
// maxSuspendFraction 0.25 (the last two are safety invariants, not knobs).
import { fileURLToPath } from 'node:url';
import { AGENTS } from './simulator.js';
import { PheromoneRouter } from './pheromone.js';
import { mulberry32 } from './prng.js';
import { runOne, summarize } from './experiment.js';

export function gridConfigs() {
  const out = [];
  for (const emaDecay of [0.5, 0.7, 0.85, 0.95]) {
    for (const pruningFactor of [0.4, 0.6, 0.8]) {
      for (const explorationRate of [0.05, 0.1]) {
        for (const roleMode of ['per-agent', 'shared']) {
          out.push({ emaDecay, pruningFactor, explorationRate, roleMode });
        }
      }
    }
  }
  return out;
}

export function scoreConfig(mode, cfg, { seeds = 8, T = 2000 } = {}) {
  const driftPoint = Math.floor(T / 2);
  let total = 0;
  for (let seed = 1; seed <= seeds; seed++) {
    const router = new PheromoneRouter(AGENTS, { rng: mulberry32(seed * 7919 + 9), mode, name: 'grid', ...cfg });
    total += summarize(runOne(seed, T, driftPoint, [router])).grid.successRate;
  }
  return total / seeds;
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  for (const mode of ['gate', 'select']) {
    const ranked = gridConfigs()
      .map((cfg) => ({ cfg, score: scoreConfig(mode, cfg) }))
      .sort((a, b) => b.score - a.score);
    console.log(`\nmode=${mode}: ${ranked.length} configs x 8 seeds x 2000 tasks`);
    for (const { cfg, score } of ranked.slice(0, 5)) console.log(`  ${score.toFixed(4)}  ${JSON.stringify(cfg)}`);
    console.log(`  worst ${ranked.at(-1).score.toFixed(4)}  ${JSON.stringify(ranked.at(-1).cfg)}`);
  }
}
