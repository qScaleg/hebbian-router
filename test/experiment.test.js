import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runOne, runExperiment, buildRouters } from '../src/experiment.js';

test('runOne produces a reward/regret series of the right length for every router', () => {
  const T = 200;
  const { series } = runOne(1, T, Math.floor(T / 2));
  const names = buildRouters(1).map((r) => r.name);
  assert.deepEqual(Object.keys(series).sort(), names.sort());
  for (const name of names) {
    assert.equal(series[name].rewards.length, T);
    assert.equal(series[name].regrets.length, T);
    for (const r of series[name].rewards) assert.ok(r === 0 || r === 1);
  }
});

test('regret is always non-negative (oracle is defined as the max-probability agent)', () => {
  const { series } = runOne(2, 300, 150);
  for (const { regrets } of Object.values(series)) {
    for (const r of regrets) assert.ok(r >= -1e-9, `negative regret ${r}`);
  }
});

test('runOne is deterministic for a fixed seed', () => {
  const a = runOne(5, 100, 50);
  const b = runOne(5, 100, 50);
  assert.deepEqual(a.series, b.series);
});

test('runExperiment aggregates across seeds and returns a sane summary shape', () => {
  const result = runExperiment({ seeds: 3, T: 300 });
  assert.equal(result.seeds, 3);
  assert.equal(result.T, 300);
  assert.equal(result.driftPoint, 150);
  for (const name of result.routerNames) {
    const a = result.aggregate[name];
    assert.ok(a.successRate >= 0 && a.successRate <= 1);
    assert.ok(a.totalRegret >= 0);
    assert.ok(a.preDriftSuccessRate >= 0 && a.preDriftSuccessRate <= 1);
    assert.ok(a.earlyPostDriftSuccessRate >= 0 && a.earlyPostDriftSuccessRate <= 1);
    assert.ok(a.latePostDriftSuccessRate >= 0 && a.latePostDriftSuccessRate <= 1);
  }
});

test('the static keyword router accumulates strictly more regret than a learning router over a long run', () => {
  // Over a long enough run the keyword router's blind spot (perf/refactor)
  // should cost it more cumulative regret than at least one learning router.
  const result = runExperiment({ seeds: 5, T: 1500 });
  const keywordRegret = result.aggregate['keyword'].totalRegret;
  const thompsonRegret = result.aggregate['thompson'].totalRegret;
  assert.ok(thompsonRegret < keywordRegret, `thompson (${thompsonRegret}) should beat keyword (${keywordRegret})`);
});
