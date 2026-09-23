// Grid search for BioMeshRouter variants, within the budget the Hebbian and
// pheromone routers got: at most 60 configs x 8 seeds (1-8) x 2000 tasks,
// drift at 1000. Each variant is tuned separately over the knobs it uses:
//   lr               {0.1, 0.3}      library default 0.1 (trainer.py:51)
//   temperature      {0.05, 0.1, 0.2} selection (ASSUMPTION, not in the library)
//   lam              {1, 10, 50}     library default 10 (ewc.py:31); EWC variants only
//   consolidateEvery {40, 200}       default 40 (ASSUMPTION); EWC variants only
// That is 36 configs for the EWC variants and 6 for plain SGD. Everything
// else stays at the library defaults (BIOMESH_DEFAULTS), with epsilon 0.02.
import { fileURLToPath } from 'node:url';
import { AGENTS, TASK_TYPES } from './simulator.js';
import { BioMeshRouter } from './biomesh.js';
import { mulberry32 } from './prng.js';
import { runOne, summarize } from './experiment.js';

export const BIOMESH_VARIANTS = {
  'biomesh-sgd': { consolidation: false, replay: false, neurogenesis: false },
  'biomesh-consolidation-only': { consolidation: true, replay: false, neurogenesis: false },
  'biomesh-consolidation+replay': { consolidation: true, replay: true, neurogenesis: false },
  'biomesh-full': { consolidation: true, replay: true, neurogenesis: true },
};

export function gridConfigs(variant) {
  const ewc = BIOMESH_VARIANTS[variant].consolidation;
  const out = [];
  for (const lr of [0.1, 0.3]) {
    for (const temperature of [0.05, 0.1, 0.2]) {
      if (!ewc) { out.push({ lr, temperature }); continue; }
      for (const lam of [1, 10, 50]) {
        for (const consolidateEvery of [40, 200]) out.push({ lr, temperature, lam, consolidateEvery });
      }
    }
  }
  return out;
}

export function scoreConfig(variant, cfg, { seeds = 8, T = 2000 } = {}) {
  const driftPoint = Math.floor(T / 2);
  let total = 0;
  for (let seed = 1; seed <= seeds; seed++) {
    const router = new BioMeshRouter(AGENTS, TASK_TYPES, {
      rng: mulberry32(seed * 7919 + 13), name: 'grid', ...BIOMESH_VARIANTS[variant], ...cfg,
    });
    total += summarize(runOne(seed, T, driftPoint, [router])).grid.successRate;
  }
  return total / seeds;
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  for (const variant of Object.keys(BIOMESH_VARIANTS)) {
    const ranked = gridConfigs(variant)
      .map((cfg) => ({ cfg, score: scoreConfig(variant, cfg) }))
      .sort((a, b) => b.score - a.score);
    console.log(`\n${variant}: ${ranked.length} configs x 8 seeds x 2000 tasks`);
    for (const { cfg, score } of ranked.slice(0, 4)) console.log(`  ${score.toFixed(4)}  ${JSON.stringify(cfg)}`);
    console.log(`  worst ${ranked.at(-1).score.toFixed(4)}  ${JSON.stringify(ranked.at(-1).cfg)}`);
  }
}
