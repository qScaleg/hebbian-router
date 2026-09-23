import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/prng.js';
import { createWorld, AGENTS, TASK_TYPES, KEYWORDS_BY_TYPE } from '../src/simulator.js';

test('sampleTask returns a known type and non-empty keywords', () => {
  const world = createWorld();
  const rng = mulberry32(1);
  for (let i = 0; i < 100; i++) {
    const task = world.sampleTask(rng, i);
    assert.ok(TASK_TYPES.includes(task.type));
    assert.ok(task.keywords.length >= 2);
    for (const w of task.keywords) {
      const allWords = Object.values(KEYWORDS_BY_TYPE).flat();
      assert.ok(allWords.includes(w));
    }
  }
});

test('oracleAgent picks the obvious best agent for obvious task types', () => {
  const world = createWorld();
  assert.equal(world.oracleAgent('security', 0), 'security-auditor');
  assert.equal(world.oracleAgent('api', 0), 'coder');
  assert.equal(world.oracleAgent('testing', 0), 'tester');
  assert.equal(world.oracleAgent('docs', 0), 'docs-writer');
});

test('oracleAgent reveals the non-obvious pairings before drift', () => {
  const world = createWorld();
  // perf and refactor are the deliberately non-obvious traps: the
  // keyword-mapped agent (coder) is NOT the hidden best agent.
  assert.equal(world.oracleAgent('perf', 0), 'architect');
  assert.equal(world.oracleAgent('refactor', 0), 'reviewer');
});

test('drift flips the oracle agent for perf and refactor after driftPoint', () => {
  const world = createWorld({ driftPoint: 100 });
  assert.equal(world.oracleAgent('perf', 99), 'architect');
  assert.equal(world.oracleAgent('perf', 100), 'coder');
  assert.equal(world.oracleAgent('refactor', 99), 'reviewer');
  assert.equal(world.oracleAgent('refactor', 100), 'architect');
  // unaffected types stay stable across the drift boundary
  assert.equal(world.oracleAgent('security', 100), 'security-auditor');
});

test('successProbability is a valid probability for every type/agent pair', () => {
  const world = createWorld({ driftPoint: 50 });
  for (const index of [0, 49, 50, 200]) {
    for (const type of TASK_TYPES) {
      for (const agent of AGENTS) {
        const p = world.successProbability(type, agent, index);
        assert.ok(p >= 0 && p <= 1, `${type}/${agent}@${index} = ${p}`);
      }
    }
  }
});

test('resolveOutcome is deterministic given the same rng state', () => {
  const world = createWorld();
  const task = { index: 0, type: 'api', keywords: ['api'], text: 'api' };
  const r1 = mulberry32(21);
  const r2 = mulberry32(21);
  const outcomes1 = Array.from({ length: 50 }, () => world.resolveOutcome(r1, task, 'coder'));
  const outcomes2 = Array.from({ length: 50 }, () => world.resolveOutcome(r2, task, 'coder'));
  assert.deepEqual(outcomes1, outcomes2);
});
