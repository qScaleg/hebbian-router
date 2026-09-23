// Synthetic world: task types, agents, hidden success probabilities, and
// noisy keyword text generation. Routers only ever see `task.type` (a
// declared category, like a ticket label) and `task.keywords`/`task.text`
// (free text) -- never the hidden probability table directly.
import { choice, bernoulli, randInt, shuffle } from './prng.js';

export const AGENTS = [
  'coder',
  'tester',
  'reviewer',
  'security-auditor',
  'architect',
  'docs-writer',
];

export const TASK_TYPES = ['security', 'api', 'testing', 'docs', 'perf', 'refactor'];

// Hidden success-probability table: BASE_PROBS[type][agent] = P(success).
// `security`, `api`, `testing`, `docs` reward the "obvious" keyword-mapped
// agent. `perf` and `refactor` are deliberately non-obvious: the
// keyword-mapped agent (coder, see KEYWORD_AGENT_MAP) is mediocre while a
// different agent hidden here is actually best -- this is the trap that a
// pure keyword router cannot see but an outcome-learning router can.
const BASE_PROBS = {
  security: { coder: 0.35, tester: 0.30, reviewer: 0.55, 'security-auditor': 0.85, architect: 0.40, 'docs-writer': 0.15 },
  api: { coder: 0.80, tester: 0.35, reviewer: 0.45, 'security-auditor': 0.30, architect: 0.55, 'docs-writer': 0.20 },
  testing: { coder: 0.40, tester: 0.82, reviewer: 0.50, 'security-auditor': 0.25, architect: 0.30, 'docs-writer': 0.15 },
  docs: { coder: 0.30, tester: 0.20, reviewer: 0.35, 'security-auditor': 0.15, architect: 0.25, 'docs-writer': 0.80 },
  perf: { coder: 0.45, tester: 0.20, reviewer: 0.40, 'security-auditor': 0.15, architect: 0.75, 'docs-writer': 0.10 },
  refactor: { coder: 0.50, tester: 0.25, reviewer: 0.78, 'security-auditor': 0.20, architect: 0.45, 'docs-writer': 0.15 },
};

// After drift, agents' hidden skills change: perf's best agent flips from
// architect to coder, and refactor's best agent flips from reviewer to
// architect. This models real skill drift (people learn/forget, models get
// swapped) and is used to test adaptation speed.
//
// Deliberately mild: the previously-best agent stays decent (architect for
// perf, reviewer for refactor) rather than collapsing to failure. This is
// on purpose -- a router that only ever unlearns via negative outcomes on
// the agent it keeps picking will "accidentally" recover quickly whenever
// the old favorite starts failing outright. A milder decline (old favorite
// still succeeds fairly often) isolates the actual thing weight decay is
// for: forgetting a stale-but-still-okay association fast enough to let a
// newly-superior option win on its own merits, not because the old one blew
// up.
const DRIFTED_PROBS = {
  ...BASE_PROBS,
  perf: { coder: 0.75, tester: 0.20, reviewer: 0.40, 'security-auditor': 0.15, architect: 0.55, 'docs-writer': 0.10 },
  refactor: { coder: 0.50, tester: 0.25, reviewer: 0.55, 'security-auditor': 0.20, architect: 0.82, 'docs-writer': 0.15 },
};

export const KEYWORDS_BY_TYPE = {
  security: ['security', 'vulnerability', 'exploit', 'auth', 'encrypt'],
  api: ['api', 'endpoint', 'rest', 'graphql', 'route'],
  testing: ['test', 'spec', 'coverage', 'assert', 'mock'],
  docs: ['docs', 'readme', 'document', 'guide', 'tutorial'],
  perf: ['performance', 'slow', 'optimize', 'latency', 'bottleneck'],
  refactor: ['refactor', 'cleanup', 'restructure', 'rename', 'simplify'],
};

// Static keyword-router type->agent map (mirrors a ruflo-style default
// routing table: a fixed lookup with no learning).
export const KEYWORD_AGENT_MAP = {
  security: 'security-auditor',
  api: 'coder',
  testing: 'tester',
  docs: 'docs-writer',
  perf: 'coder',
  refactor: 'coder',
};

/** Build a flat word -> agent lookup from KEYWORDS_BY_TYPE + KEYWORD_AGENT_MAP. */
export function wordAgentMap() {
  const map = {};
  for (const [type, words] of Object.entries(KEYWORDS_BY_TYPE)) {
    const agent = KEYWORD_AGENT_MAP[type];
    for (const w of words) map[w] = agent;
  }
  return map;
}

function sampleKeywords(rng, type) {
  const own = KEYWORDS_BY_TYPE[type];
  const n = 2 + randInt(rng, 2); // 2-3 own keywords
  const words = new Set();
  while (words.size < n) words.add(choice(rng, own));
  // 25% of tasks borrow one keyword from an unrelated type -- realistic
  // noisy text (e.g. "docs for the api endpoint") that can mislead a
  // pure keyword matcher.
  if (bernoulli(rng, 0.25)) {
    let otherType = choice(rng, TASK_TYPES);
    if (otherType === type) otherType = TASK_TYPES[(TASK_TYPES.indexOf(type) + 1) % TASK_TYPES.length];
    words.add(choice(rng, KEYWORDS_BY_TYPE[otherType]));
  }
  return shuffle(rng, [...words]);
}

/**
 * Create a synthetic world. `driftPoint`, if set, is the task index at
 * which hidden skills switch from BASE_PROBS to DRIFTED_PROBS.
 */
export function createWorld({ driftPoint = null } = {}) {
  function probsAt(index) {
    return driftPoint != null && index >= driftPoint ? DRIFTED_PROBS : BASE_PROBS;
  }
  return {
    agents: AGENTS,
    taskTypes: TASK_TYPES,
    driftPoint,

    sampleTask(rng, index) {
      const type = choice(rng, TASK_TYPES);
      const keywords = sampleKeywords(rng, type);
      return { index, type, keywords, text: keywords.join(' ') };
    },

    successProbability(type, agent, index) {
      return probsAt(index)[type][agent];
    },

    oracleAgent(type, index) {
      const probs = probsAt(index)[type];
      return AGENTS.reduce((best, a) => (probs[a] > probs[best] ? a : best), AGENTS[0]);
    },

    resolveOutcome(rng, task, agent) {
      const p = this.successProbability(task.type, agent, task.index);
      return bernoulli(rng, p);
    },
  };
}
