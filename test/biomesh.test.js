import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/prng.js';
import { AGENTS, TASK_TYPES, createWorld } from '../src/simulator.js';
import { runOne } from '../src/experiment.js';
import { BioMeshRouter, EpisodicBuffer, LongTermStore, sleepCycle } from '../src/biomesh.js';

function drive(router, T, seed, onStep = () => {}) {
  const world = createWorld({ driftPoint: Math.floor(T / 2) });
  const taskRng = mulberry32(seed);
  const outRng = mulberry32(seed + 1);
  for (let i = 0; i < T; i++) {
    const task = world.sampleTask(taskRng, i);
    const agent = router.select(task);
    const reward = outRng() < world.successProbability(task.type, agent, i) ? 1 : 0;
    router.update(task, agent, reward);
    onStep(i);
  }
}

test('BioMeshRouter is deterministic for a fixed seed', () => {
  const make = () => [new BioMeshRouter(AGENTS, TASK_TYPES, { rng: mulberry32(3), name: 'b' })];
  assert.deepEqual(runOne(6, 600, 300, make()).series, runOne(6, 600, 300, make()).series);
});

test('frozen weights never change once frozen (EWC freeze and pruning)', () => {
  const router = new BioMeshRouter(AGENTS, TASK_TYPES, { rng: mulberry32(4), name: 'b' });
  const frozenAt = new Map();
  drive(router, 2000, 10, () => {
    for (const p of router.params) {
      if (frozenAt.has(p)) {
        assert.equal(p.frozen, true, 'a frozen parameter was unfrozen');
        assert.equal(p.value, frozenAt.get(p), 'a frozen parameter moved');
      } else if (p.frozen) frozenAt.set(p, p.value);
    }
  });
  assert.ok(frozenAt.size > 0, 'test must actually exercise freezing');
});

test('growth fires only after the probe patience window', () => {
  // errorThreshold -1 makes the probe-loss trigger always true, so the only
  // thing holding growth back is the patience window.
  const probeSteps = 10;
  const router = new BioMeshRouter(AGENTS, ['perf'], {
    rng: mulberry32(5), name: 'b', errorThreshold: -1, probeSteps, consolidateEvery: 40, maxLayers: 8,
  });
  const task = { type: 'perf', keywords: ['performance'] };
  for (let i = 0; i < 200; i++) {
    const before = router.rows.get('perf').layers.length;
    router.update(task, router.select(task), 1);
    const grew = router.rows.get('perf').layers.length > before;
    const posInSegment = (i % 40) + 1; // observations of the row this segment
    assert.equal(grew, posInSegment === probeSteps && before < 8, `step ${i}`);
  }
  for (const e of router.log.filter((x) => x.event === 'grow')) assert.equal(e.segObs, probeSteps);
});

test('growth never exceeds maxLayers', () => {
  const router = new BioMeshRouter(AGENTS, ['perf'], { rng: mulberry32(6), name: 'b', errorThreshold: -1, maxLayers: 3 });
  const task = { type: 'perf', keywords: [] };
  for (let i = 0; i < 1000; i++) router.update(task, router.select(task), 0);
  assert.equal(router.rows.get('perf').layers.length, 3);
});

test('replay never exceeds the buffer contents or the batch size', () => {
  const rng = mulberry32(7);
  for (let trial = 0; trial < 200; trial++) {
    const buffer = new EpisodicBuffer(128);
    const store = new LongTermStore(64);
    // Some older segments already in long-term memory.
    const oldSegments = Math.floor(rng() * 4);
    for (let s = 0; s < oldSegments; s++) {
      store.integrate(Array.from({ length: 1 + Math.floor(rng() * 5) }, (_, k) => ({ task: s, id: `old${s}-${k}` })), rng);
    }
    const k = Math.floor(rng() * 20);
    const pushed = Array.from({ length: k }, (_, i) => ({ task: 99, id: `new${i}` }));
    for (const e of pushed) buffer.push(e);
    const storeBefore = store.length;
    const { batch, recentCount } = sleepCycle(buffer, store, rng, { replayBatch: 16, replayRatio: 0.5 }, 99);
    const fresh = batch.filter((e) => e.task === 99);
    const old = batch.filter((e) => e.task !== 99);
    assert.equal(recentCount, k);
    assert.ok(batch.length <= 16);
    assert.ok(fresh.length <= k, 'replayed more recent items than the buffer held');
    assert.equal(new Set(fresh.map((e) => e.id)).size, fresh.length, 'recent items sampled with replacement');
    for (const e of fresh) assert.ok(pushed.includes(e));
    assert.ok(old.length <= Math.min(8, storeBefore));
    assert.equal(buffer.length, 0, 'sleep must drain the buffer');
  }
});

test('long-term store keeps at most perTaskCapacity per task and samples every old task', () => {
  const rng = mulberry32(8);
  const store = new LongTermStore(5);
  for (let t = 0; t < 3; t++) store.integrate(Array.from({ length: 50 }, (_, i) => ({ task: t, i })), rng);
  assert.equal(store.length, 15);
  const s = store.sample(6, rng, 2);
  assert.deepEqual([...new Set(s.map((e) => e.task))].sort(), [0, 1]);
  assert.equal(s.length, 6);
});

test('every variant converges to the oracle agent in a trivial world', () => {
  const variants = {
    'consolidation-only': { replay: false, neurogenesis: false },
    'consolidation+replay': { neurogenesis: false },
    full: {},
  };
  for (const [name, cfg] of Object.entries(variants)) {
    const router = new BioMeshRouter(AGENTS, ['perf'], { rng: mulberry32(9), name, temperature: 0.1, ...cfg });
    const task = { type: 'perf', keywords: ['performance'] };
    let late = 0;
    for (let i = 0; i < 600; i++) {
      const agent = router.select(task);
      router.update(task, agent, agent === 'architect' ? 1 : 0);
      if (i >= 400 && agent === 'architect') late += 1;
    }
    assert.ok(late >= 180, `${name}: expected near-certain convergence, got ${late}/200`);
  }
});
