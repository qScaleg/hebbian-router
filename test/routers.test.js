import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/prng.js';
import { AGENTS, TASK_TYPES } from '../src/simulator.js';
import { KeywordRouter, ThompsonRouter, HebbianRouter } from '../src/routers.js';

// --- KeywordRouter ---------------------------------------------------

test('KeywordRouter maps an unambiguous keyword to its static agent', () => {
  const router = new KeywordRouter(AGENTS);
  assert.equal(router.select({ keywords: ['security'] }), 'security-auditor');
  assert.equal(router.select({ keywords: ['test', 'assert'] }), 'tester');
  assert.equal(router.select({ keywords: ['readme', 'guide'] }), 'docs-writer');
});

test('KeywordRouter falls back when no keyword matches', () => {
  const router = new KeywordRouter(AGENTS, { fallback: 'coder' });
  assert.equal(router.select({ keywords: ['unknownword'] }), 'coder');
});

test('KeywordRouter demonstrates the perf/refactor trap: it never picks the hidden-best agent', () => {
  const router = new KeywordRouter(AGENTS);
  // perf keywords route to coder, but the hidden oracle for perf is architect.
  assert.equal(router.select({ keywords: ['performance', 'slow'] }), 'coder');
  // refactor keywords route to coder, but the hidden oracle is reviewer.
  assert.equal(router.select({ keywords: ['refactor', 'cleanup'] }), 'coder');
});

// --- ThompsonRouter ----------------------------------------------------

test('ThompsonRouter picks the oracle agent in a trivial world after training', () => {
  const rng = mulberry32(1);
  const router = new ThompsonRouter(AGENTS, TASK_TYPES, rng);
  const task = { type: 'perf', keywords: ['performance'] };
  // Train: architect always succeeds, everyone else always fails.
  for (let i = 0; i < 200; i++) {
    for (const agent of AGENTS) {
      router.update(task, agent, agent === 'architect' ? 1 : 0);
    }
  }
  let architectPicks = 0;
  for (let i = 0; i < 200; i++) {
    if (router.select(task) === 'architect') architectPicks += 1;
  }
  assert.ok(architectPicks > 190, `expected near-certain convergence, got ${architectPicks}/200`);
});

// --- HebbianRouter -------------------------------------------------------

test('HebbianRouter learns to score the oracle agent highest in a trivial world', () => {
  const rng = mulberry32(2);
  const router = new HebbianRouter(AGENTS, { rng, decay: 0.02, epsilon: 0 });
  const task = { type: 'refactor', keywords: ['refactor', 'cleanup'] };
  for (let i = 0; i < 500; i++) {
    for (const agent of AGENTS) {
      router.update(task, agent, agent === 'reviewer' ? 1 : 0);
    }
  }
  const scores = router.scores(task);
  const best = AGENTS.reduce((a, b) => (scores[b] > scores[a] ? b : a));
  assert.equal(best, 'reviewer');
});

test('HebbianRouter with epsilon=0 mostly selects the learned-best agent', () => {
  const rng = mulberry32(3);
  const router = new HebbianRouter(AGENTS, { rng, decay: 0.02, epsilon: 0, temperature: 0.1 });
  const task = { type: 'docs', keywords: ['readme'] };
  for (let i = 0; i < 500; i++) {
    for (const agent of AGENTS) {
      router.update(task, agent, agent === 'docs-writer' ? 1 : 0);
    }
  }
  let picks = 0;
  for (let i = 0; i < 100; i++) {
    if (router.select(task) === 'docs-writer') picks += 1;
  }
  assert.ok(picks > 90, `expected near-greedy selection, got ${picks}/100`);
});

test('meshStyle (mesh.rs-mirroring) no-decay variant saturates at the weight cap; decayed variant settles below it', () => {
  const rngNoDecay = mulberry32(4);
  const rngDecay = mulberry32(5);
  const noDecay = new HebbianRouter(AGENTS, { rng: rngNoDecay, meshStyle: true, eta: 0.5 });
  const decayed = new HebbianRouter(AGENTS, { rng: rngDecay, decay: 0.1, eta: 0.5 });
  const task = { type: 'security', keywords: ['security'] };
  for (let i = 0; i < 1000; i++) {
    noDecay.update(task, 'security-auditor', 1);
    decayed.update(task, 'security-auditor', 1);
  }
  const wNoDecay = noDecay.weightsFor('security').get('security-auditor');
  const wDecayed = decayed.weightsFor('security').get('security-auditor');
  assert.ok(wNoDecay > 0.95, `no-decay weight should saturate near cap, got ${wNoDecay}`);
  assert.ok(wDecayed < wNoDecay - 0.05, `decayed weight (${wDecayed}) should settle below no-decay (${wNoDecay})`);
});

test('meshStyle never decreases a weight on failure (strengthen-only, mirroring mesh.rs); the default rule does', () => {
  const meshRouter = new HebbianRouter(AGENTS, { rng: mulberry32(8), meshStyle: true, eta: 0.5 });
  const defaultRouter = new HebbianRouter(AGENTS, { rng: mulberry32(9), decay: 0, eta: 0.5 });
  const task = { type: 'perf', keywords: ['performance'] };
  // Build up some weight with a few successes, then hit it with failures.
  for (let i = 0; i < 10; i++) {
    meshRouter.update(task, 'coder', 1);
    defaultRouter.update(task, 'coder', 1);
  }
  const meshBefore = meshRouter.weightsFor('perf').get('coder');
  const defaultBefore = defaultRouter.weightsFor('perf').get('coder');
  for (let i = 0; i < 10; i++) {
    meshRouter.update(task, 'coder', 0);
    defaultRouter.update(task, 'coder', 0);
  }
  const meshAfter = meshRouter.weightsFor('perf').get('coder');
  const defaultAfter = defaultRouter.weightsFor('perf').get('coder');
  assert.ok(meshAfter >= meshBefore - 1e-9, `meshStyle weight should never drop on failure: ${meshBefore} -> ${meshAfter}`);
  assert.ok(defaultAfter < defaultBefore, `default rule should punish repeated failure: ${defaultBefore} -> ${defaultAfter}`);
});

test('meshStyle variant is named hebbian-no-decay and default variant is named hebbian', () => {
  const a = new HebbianRouter(AGENTS, { rng: mulberry32(6), meshStyle: true });
  const b = new HebbianRouter(AGENTS, { rng: mulberry32(7), decay: 0.02 });
  assert.equal(a.name, 'hebbian-no-decay');
  assert.equal(a.decay, 0); // meshStyle forces decay off regardless of what's passed
  assert.equal(b.name, 'hebbian');
});
