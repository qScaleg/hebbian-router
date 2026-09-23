// ruflo's `pheromone-adaptive` topology (ADR-330, "APSC"), ported to plain JS
// so it can run in this simulation.
//
// Source of truth (ruflo 3.42.5 as installed globally):
//   ruflo/node_modules/@claude-flow/cli/dist/src/services/pheromone-adaptive.js
//   (TS source: ruflo/v3/@claude-flow/cli/src/services/pheromone-adaptive.ts)
//   ruflo/v3/docs/adr/ADR-330-adaptive-pheromone-swarm-consensus.md
//
// `recordApscSignal` below is a line-for-line port of ruflo's function of the
// same name. The only intentional differences:
//   - it mutates `state` in place instead of deep-copying it every call
//     (same results, less garbage);
//   - `lastUpdatedAt` is the round number rather than an ISO wall-clock
//     timestamp, so runs are deterministic. ruflo only uses it to order
//     suspended agents for exploration ("stalest first"), and round order is
//     the same order.
//
// What APSC is NOT: it is not an agent selector. In ruflo it is a scheduling
// eligibility gate (`isApscAgentEligible`, used by agent_execute in
// mcp-tools/agent-tools.js): whatever picked the agent, a live APSC swarm
// refuses dispatch to a suspended one. Default config is dry-run, which never
// refuses anything. See PheromoneRouter below for how that maps onto routing.
import { KeywordRouter } from './routers.js';

// pheromone-adaptive.ts DEFAULT_APSC_CONFIG (dist .js lines 12-24).
export const DEFAULT_APSC_CONFIG = Object.freeze({
  alpha: 0.5, // weight on taskSuccess
  beta: 0.2, // weight on (1 - normalizedLatency)
  gamma: 0.3, // weight on consensusAlignment
  emaDecay: 0.85, // agent EMA, role baseline EMA and threshold EMA all use it
  pruningFactor: 0.6, // suspend if ema < threshold * pruningFactor
  reactivationThreshold: 0.75, // reactivate if ema >= threshold * this
  minActiveAgents: 3, // quorum floor
  minSamples: 3, // no pruning before this many observations
  maxSuspendFraction: 0.25, // floor(active * this) suspensions allowed per round
  explorationRate: 0.1, // every round(1/rate) rounds, reactivate stalest suspended
  dryRun: true, // ruflo default; live needs --apsc-live
  protectedRoles: ['coordinator', 'queen', 'security-architect', 'security-auditor'],
});

function unit(value, field) {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${field} must be a finite number in [0,1]`);
  }
  return value;
}

// Port of normalizeApscConfig: validates and renormalises alpha/beta/gamma to sum to 1.
export function normalizeApscConfig(input = {}) {
  const d = DEFAULT_APSC_CONFIG;
  const alpha = unit(input.alpha ?? d.alpha, 'alpha');
  const beta = unit(input.beta ?? d.beta, 'beta');
  const gamma = unit(input.gamma ?? d.gamma, 'gamma');
  const weight = alpha + beta + gamma;
  if (weight <= 0) throw new Error('at least one APSC score weight must be positive');
  const minActiveAgents = input.minActiveAgents ?? d.minActiveAgents;
  const minSamples = input.minSamples ?? d.minSamples;
  if (!Number.isInteger(minActiveAgents) || minActiveAgents < 1 || minActiveAgents > 50) {
    throw new Error('minActiveAgents must be an integer in [1,50]');
  }
  if (!Number.isInteger(minSamples) || minSamples < 1 || minSamples > 1000) {
    throw new Error('minSamples must be an integer in [1,1000]');
  }
  const protectedRoles = [...new Set((input.protectedRoles ?? d.protectedRoles)
    .map((r) => r.trim().toLowerCase()).filter(Boolean))].sort();
  return {
    alpha: alpha / weight,
    beta: beta / weight,
    gamma: gamma / weight,
    emaDecay: unit(input.emaDecay ?? d.emaDecay, 'emaDecay'),
    pruningFactor: unit(input.pruningFactor ?? d.pruningFactor, 'pruningFactor'),
    reactivationThreshold: unit(input.reactivationThreshold ?? d.reactivationThreshold, 'reactivationThreshold'),
    minActiveAgents,
    minSamples,
    maxSuspendFraction: unit(input.maxSuspendFraction ?? d.maxSuspendFraction, 'maxSuspendFraction'),
    explorationRate: unit(input.explorationRate ?? d.explorationRate, 'explorationRate'),
    dryRun: input.dryRun ?? d.dryRun,
    protectedRoles,
  };
}

export function createApscState(config = {}) {
  // Initial threshold 0.5: createApscState in pheromone-adaptive.ts.
  return { config: normalizeApscConfig(config), round: 0, threshold: 0.5, roleBaselines: {}, agents: {} };
}

export function apscCounts(state) {
  let activeAgents = 0;
  let suspendedAgents = 0;
  for (const a of Object.values(state.agents)) {
    if (a.status === 'active') activeAgents += 1;
    else suspendedAgents += 1;
  }
  return { activeAgents, suspendedAgents };
}

function isProtected(state, role) {
  return state.config.protectedRoles.includes(role.toLowerCase());
}

/** Port of ruflo's recordApscSignal. Mutates `state`; returns the decision. */
export function recordApscSignal(state, signal) {
  const config = state.config;
  state.round += 1;
  const success = unit(signal.taskSuccess, 'taskSuccess');
  const latency = unit(signal.normalizedLatency, 'normalizedLatency');
  const alignment = unit(signal.consensusAlignment, 'consensusAlignment');
  const rawScore = config.alpha * success + config.beta * (1 - latency) + config.gamma * alignment;

  // Role normalisation: centre on the mean latest rawScore of *other* agents
  // in the same role, else the role's EMA baseline, else this score itself.
  const role = signal.role.toLowerCase();
  let peerSum = 0;
  let peerN = 0;
  for (const item of Object.values(state.agents)) {
    if (item.agentId !== signal.agentId && item.role === role && item.samples > 0) {
      peerSum += item.rawScore;
      peerN += 1;
    }
  }
  const priorRoleBaseline = peerN ? peerSum / peerN : (state.roleBaselines[role] ?? rawScore);
  const normalizedScore = Math.max(0, Math.min(1, 0.5 + rawScore - priorRoleBaseline));

  const previous = state.agents[signal.agentId];
  const agent = {
    agentId: signal.agentId,
    role,
    status: previous?.status ?? 'active',
    samples: (previous?.samples ?? 0) + 1,
    rawScore,
    normalizedScore,
    emaScore: previous
      ? config.emaDecay * previous.emaScore + (1 - config.emaDecay) * normalizedScore
      : normalizedScore,
    lastUpdatedAt: state.round,
    lastDecision: 'keep',
  };
  state.agents[signal.agentId] = agent;

  let roleSum = 0;
  let roleN = 0;
  for (const item of Object.values(state.agents)) {
    if (item.role === role) { roleSum += item.rawScore; roleN += 1; }
  }
  state.roleBaselines[role] = config.emaDecay * priorRoleBaseline + (1 - config.emaDecay) * (roleSum / roleN);

  // Adaptive threshold: EMA of the mean emaScore over mature agents.
  let matureSum = 0;
  let matureN = 0;
  for (const item of Object.values(state.agents)) {
    if (item.samples >= config.minSamples) { matureSum += item.emaScore; matureN += 1; }
  }
  const observedMean = matureN ? matureSum / matureN : state.threshold;
  state.threshold = config.emaDecay * state.threshold + (1 - config.emaDecay) * observedMean;

  let action = 'keep';
  let applied = false;
  if (agent.status === 'suspended' && agent.emaScore >= state.threshold * config.reactivationThreshold) {
    action = 'reactivate';
    if (!config.dryRun) { agent.status = 'active'; applied = true; }
  } else if (
    agent.status === 'active'
    && agent.samples >= config.minSamples
    && !isProtected(state, agent.role)
    && agent.emaScore < state.threshold * config.pruningFactor
  ) {
    const before = apscCounts(state);
    const maxThisRound = Math.floor(before.activeAgents * config.maxSuspendFraction);
    const quorumCapacity = Math.max(0, before.activeAgents - config.minActiveAgents);
    if (maxThisRound >= 1 && quorumCapacity >= 1) {
      action = config.dryRun ? 'would-suspend' : 'suspend';
      if (!config.dryRun) { agent.status = 'suspended'; applied = true; }
    }
  }

  // Deterministic exploration: every round(1/explorationRate) rounds,
  // reactivate the stalest suspended agent (ties broken by agentId).
  if (!config.dryRun && config.explorationRate > 0) {
    const cadence = Math.max(1, Math.round(1 / config.explorationRate));
    if (state.round % cadence === 0) {
      const explorer = Object.values(state.agents)
        .filter((item) => item.status === 'suspended')
        .sort((a, b) => a.lastUpdatedAt - b.lastUpdatedAt || a.agentId.localeCompare(b.agentId))[0];
      if (explorer) {
        explorer.status = 'active';
        explorer.lastDecision = 'explore';
        if (explorer.agentId === agent.agentId) { action = 'explore'; applied = true; }
      }
    }
  }
  agent.lastDecision = action;
  return { action, applied, emaScore: agent.emaScore, threshold: state.threshold, ...apscCounts(state) };
}

/** Port of isApscAgentEligible: dry-run and unknown agents are always eligible. */
export function isApscAgentEligible(state, agentId) {
  if (!state || state.config.dryRun) return true;
  return state.agents[agentId]?.status !== 'suspended';
}

// --- PheromoneRouter --------------------------------------------------------
//
// Mapping the simulator onto APSC signals:
//   taskSuccess        = the task's 0/1 reward (the only outcome signal the
//                        simulator has).
//   normalizedLatency  = NEUTRAL 0.5 constant. The simulator has no latency.
//                        (ruflo's hooks_post-task derives it from duration /
//                        latencyBudgetMs.)
//   consensusAlignment = NEUTRAL 0.5 constant. There is no consensus step in
//                        this simulation. (hooks_post-task defaults it to the
//                        task's `quality`.)
// Both constants add the same amount to every rawScore, so role-centring
// cancels them out apart from clamp effects: only success moves fitness.
//
// Roles: agentId = the simulator's agent name. `roleMode: 'per-agent'` uses the
// agent name as its role, which is what hooks_post-task does when no
// agentRole is passed (`role: String(params.agentRole ?? agent)`). With one
// agent per role, each agent is centred on its OWN baseline, so fitness
// tracks "better or worse than this agent's recent self" rather than "better
// than other agents". `roleMode: 'shared'` gives every agent one role
// ('worker') so they are centred on each other; this is a legal ruflo config
// (pass the same agentRole everywhere) but not the default.
//
// Modes:
//   'gate'   FAITHFUL. ruflo's actual behaviour: some other router picks the
//            agent (here, KeywordRouter, the static routing ruflo uses) and
//            APSC only refuses suspended agents. One APSC state for the whole
//            swarm, as in ruflo (state is per swarm, keyed by agentId, with
//            no notion of task type).
//            ASSUMPTION: when the keyword pick is suspended, ruflo's
//            agent_execute just returns an error; we re-route to the
//            eligible agent with the highest emaScore (unseen agents count
//            as the initial threshold 0.5).
//   'select' EXTENSION, not ruflo behaviour: routes directly on pheromone
//            fitness. ASSUMPTIONS: one APSC state per task type (ruflo has
//            no task context, but without it a context-free per-agent score
//            cannot learn "architect for perf"); agents below minSamples are
//            tried first (uniformly at random); otherwise argmax emaScore over
//            eligible agents, random tie-break. No extra exploration beyond
//            APSC's own reactivation cadence.
export class PheromoneRouter {
  constructor(agents, {
    mode = 'gate',
    roleMode = 'per-agent',
    rng,
    name,
    ...apsc
  } = {}) {
    if (mode !== 'gate' && mode !== 'select') throw new Error(`unknown mode ${mode}`);
    if (roleMode !== 'per-agent' && roleMode !== 'shared') throw new Error(`unknown roleMode ${roleMode}`);
    this.agents = agents;
    this.mode = mode;
    this.roleMode = roleMode;
    this.rng = rng;
    // Live mode by default: ruflo's dry-run default never denies dispatch, so
    // a dry-run PheromoneRouter('gate') is exactly KeywordRouter.
    this.apscConfig = { dryRun: false, ...apsc };
    this.base = new KeywordRouter(agents);
    this.states = new Map(); // context key -> APSC state
    this.name = name ?? (mode === 'gate' ? 'pheromone' : 'pheromone-select');
  }

  contextKey(task) {
    return this.mode === 'gate' ? '*' : task.type;
  }

  stateFor(task) {
    const key = this.contextKey(task);
    let s = this.states.get(key);
    if (!s) {
      s = createApscState(this.apscConfig);
      this.states.set(key, s);
    }
    return s;
  }

  roleOf(agent) {
    return this.roleMode === 'shared' ? 'worker' : agent;
  }

  pickRandom(list) {
    return list[Math.floor(this.rng() * list.length)];
  }

  bestByEma(state, candidates) {
    let best = [];
    let bestScore = -Infinity;
    for (const a of candidates) {
      const score = state.agents[a]?.emaScore ?? 0.5;
      if (score > bestScore + 1e-12) { best = [a]; bestScore = score; }
      else if (Math.abs(score - bestScore) <= 1e-12) best.push(a);
    }
    return best;
  }

  select(task) {
    const state = this.stateFor(task);
    const eligible = this.agents.filter((a) => isApscAgentEligible(state, a));
    if (this.mode === 'gate') {
      const pick = this.base.select(task);
      if (isApscAgentEligible(state, pick) || eligible.length === 0) return pick;
      return this.bestByEma(state, eligible)[0];
    }
    const pool = eligible.length ? eligible : this.agents; // fail open
    const warm = pool.filter((a) => (state.agents[a]?.samples ?? 0) < state.config.minSamples);
    if (warm.length) return this.pickRandom(warm);
    const best = this.bestByEma(state, pool);
    return best.length === 1 ? best[0] : this.pickRandom(best);
  }

  update(task, agent, reward) {
    return recordApscSignal(this.stateFor(task), {
      agentId: agent,
      role: this.roleOf(agent),
      taskSuccess: reward,
      normalizedLatency: 0.5,
      consensusAlignment: 0.5,
    });
  }
}
