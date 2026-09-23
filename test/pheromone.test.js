import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/prng.js';
import { AGENTS } from '../src/simulator.js';
import { KeywordRouter } from '../src/routers.js';
import { runOne } from '../src/experiment.js';
import {
  DEFAULT_APSC_CONFIG,
  PheromoneRouter,
  apscCounts,
  createApscState,
  isApscAgentEligible,
  normalizeApscConfig,
  recordApscSignal,
} from '../src/pheromone.js';

const signal = (agentId, taskSuccess, role = 'worker') => ({
  agentId, role, taskSuccess, normalizedLatency: 0.5, consensusAlignment: 0.5,
});

test('defaults match ruflo DEFAULT_APSC_CONFIG and weights renormalise to sum 1', () => {
  assert.equal(DEFAULT_APSC_CONFIG.emaDecay, 0.85);
  assert.equal(DEFAULT_APSC_CONFIG.pruningFactor, 0.6);
  assert.equal(DEFAULT_APSC_CONFIG.minActiveAgents, 3);
  assert.equal(DEFAULT_APSC_CONFIG.maxSuspendFraction, 0.25);
  assert.equal(DEFAULT_APSC_CONFIG.dryRun, true);
  const c = normalizeApscConfig({ alpha: 1, beta: 0.5, gamma: 0.5 });
  assert.equal(c.alpha, 0.5);
  assert.equal(c.alpha + c.beta + c.gamma, 1);
  assert.throws(() => normalizeApscConfig({ gamma: 2 }));
  assert.throws(() => normalizeApscConfig({ emaDecay: 1.5 }));
  assert.throws(() => normalizeApscConfig({ minActiveAgents: 0 }));
});

test('first observation of a role is centred on itself (normalizedScore 0.5)', () => {
  const s = createApscState({ dryRun: false });
  recordApscSignal(s, signal('coder', 1, 'coder'));
  assert.equal(s.agents.coder.normalizedScore, 0.5);
  assert.equal(s.agents.coder.emaScore, 0.5);
});

test('PheromoneRouter is deterministic for a fixed seed', () => {
  for (const mode of ['gate', 'select']) {
    const make = () => [new PheromoneRouter(AGENTS, { rng: mulberry32(11), mode, roleMode: 'shared', name: 'p' })];
    assert.deepEqual(runOne(4, 400, 200, make()).series, runOne(4, 400, 200, make()).series);
  }
});

test('dry-run gate never denies dispatch, so it routes exactly like KeywordRouter', () => {
  const kw = new KeywordRouter(AGENTS);
  const ph = new PheromoneRouter(AGENTS, { rng: mulberry32(1), dryRun: true, pruningFactor: 1 });
  const rng = mulberry32(2);
  for (let i = 0; i < 300; i++) {
    const task = { type: 'api', keywords: [['api', 'test', 'readme', 'slow'][i % 4]] };
    const agent = ph.select(task);
    assert.equal(agent, kw.select(task));
    ph.update(task, agent, rng() < 0.1 ? 1 : 0);
  }
});

test('select mode converges to the oracle agent in a trivial world', () => {
  const router = new PheromoneRouter(AGENTS, { rng: mulberry32(5), mode: 'select', roleMode: 'shared' });
  const task = { type: 'perf', keywords: ['performance'] };
  let late = 0;
  for (let i = 0; i < 400; i++) {
    const agent = router.select(task);
    router.update(task, agent, agent === 'architect' ? 1 : 0);
    if (i >= 200 && agent === 'architect') late += 1;
  }
  assert.ok(late >= 190, `expected near-certain convergence, got ${late}/200`);
});

test('pause/quarantine never drops active agents below the minActiveAgents floor', () => {
  for (const seed of [1, 2, 3]) {
    const rng = mulberry32(seed);
    const s = createApscState({ dryRun: false, pruningFactor: 1, minSamples: 1, explorationRate: 0 });
    const ids = Array.from({ length: 10 }, (_, i) => `a${i}`);
    let prevSuspended = 0;
    let suspensions = 0;
    for (let round = 0; round < 3000; round++) {
      const id = ids[Math.floor(rng() * ids.length)];
      if (!isApscAgentEligible(s, id)) continue; // suspended agents get no dispatch
      const before = apscCounts(s).activeAgents;
      const d = recordApscSignal(s, signal(id, rng() < 0.5 ? 1 : 0));
      const registered = Object.keys(s.agents).length;
      assert.ok(d.activeAgents >= Math.min(s.config.minActiveAgents, registered), `floor crossed: ${d.activeAgents}`);
      // maxSuspendFraction: at most floor(active * 0.25) new suspensions per round.
      const newly = Math.max(0, d.suspendedAgents - prevSuspended);
      assert.ok(newly <= Math.max(0, Math.floor((before + (registered > before ? 1 : 0)) * 0.25)));
      if (d.action === 'suspend') suspensions += 1;
      prevSuspended = d.suspendedAgents;
    }
    assert.ok(suspensions > 0, 'test must actually exercise pruning');
  }
});

test('protected roles are never paused, even when they are the worst performer', () => {
  const s = createApscState({ dryRun: false, pruningFactor: 1, minSamples: 1, explorationRate: 0 });
  const agents = [
    ['security-auditor', 'security-auditor'], // protected by ruflo default
    ['weak', 'weak-role'],
    ['a', 'x'], ['b', 'x'], ['c', 'x'], ['d', 'x'],
  ];
  let weakSuspended = false;
  for (let round = 0; round < 500; round++) {
    const [id, role] = agents[round % agents.length];
    if (!isApscAgentEligible(s, id)) continue;
    const fail = id === 'security-auditor' || id === 'weak';
    // Role-local centring would hide a uniformly weak role, so give both
    // weak agents a failure trend instead of a flat zero.
    const success = fail ? (round < 60 ? 1 : 0) : 1;
    recordApscSignal(s, { agentId: id, role, taskSuccess: success, normalizedLatency: 0.5, consensusAlignment: 0.5 });
    assert.equal(s.agents['security-auditor']?.status ?? 'active', 'active');
    if (s.agents.weak?.status === 'suspended') weakSuspended = true;
  }
  assert.ok(weakSuspended, 'an unprotected agent with the same trajectory must get suspended');
});
