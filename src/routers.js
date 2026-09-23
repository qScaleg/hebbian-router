// Three routing strategies compared by the experiment:
//   1. KeywordRouter  - static keyword->agent lookup (no learning).
//   2. ThompsonRouter - Beta-Bernoulli bandit per (task type x agent).
//   3. HebbianRouter  - reward-modulated ("three-factor") Hebbian weights
//      over task features, with a decay=0 variant mirroring the
//      no-decay update rule in Synaptic-Mesh's mesh.rs.
import { wordAgentMap } from './simulator.js';
import { sampleBeta } from './prng.js';

export function softmax(xs, temperature) {
  const scaled = xs.map((x) => x / temperature);
  const m = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

// --- 1. KeywordRouter -------------------------------------------------

export class KeywordRouter {
  constructor(agents, { fallback = 'coder' } = {}) {
    this.agents = agents;
    this.fallback = fallback;
    this.wordAgent = wordAgentMap();
    this.name = 'keyword';
  }

  select(task) {
    for (const word of task.keywords) {
      const agent = this.wordAgent[word];
      if (agent) return agent;
    }
    return this.fallback;
  }

  // Stateless: nothing to learn from outcomes.
  update() {}
}

// --- 2. ThompsonRouter --------------------------------------------------

export class ThompsonRouter {
  constructor(agents, taskTypes, rng) {
    this.agents = agents;
    this.rng = rng;
    this.alpha = {};
    this.beta = {};
    for (const t of taskTypes) {
      this.alpha[t] = Object.fromEntries(agents.map((a) => [a, 1]));
      this.beta[t] = Object.fromEntries(agents.map((a) => [a, 1]));
    }
    this.name = 'thompson';
  }

  select(task) {
    const a = this.alpha[task.type];
    const b = this.beta[task.type];
    let bestAgent = this.agents[0];
    let bestSample = -Infinity;
    for (const agent of this.agents) {
      const sample = sampleBeta(this.rng, a[agent], b[agent]);
      if (sample > bestSample) {
        bestSample = sample;
        bestAgent = agent;
      }
    }
    return bestAgent;
  }

  update(task, agent, reward) {
    if (reward) this.alpha[task.type][agent] += 1;
    else this.beta[task.type][agent] += 1;
  }
}

// --- 3. HebbianRouter -----------------------------------------------------
//
// Weights w[feature][agent] score how well a feature (task type or keyword)
// predicts success for an agent. Selection is softmax-over-scores with
// epsilon-greedy exploration. The update is a reward-modulated
// ("three-factor": pre-synaptic feature activity x post-synaptic action x
// reward signal) Hebbian rule:
//
//   delta w = eta * pre * (reward - baseline)     [only for the chosen agent]
//   w *= (1 - decay)                               [weight decay]
//   w = clamp(w, 0, 1)                             [mirrors mesh.rs's cap of 1.0]
//   row rescaled if its weight sum exceeds agents.length  [normalisation]
//
// `meshStyle: true` reproduces the Synaptic-Mesh mesh.rs update rule
// literally: strengthen-only reinforcement (contrib1*contrib2*0.1, capped at
// 1.0) with NO decay and NO negative correction on failure -- a weight can
// only ever grow or hold steady, never shrink except via the shared
// normalisation step. That is the actual defect being demonstrated: once an
// association is learned it cannot be *unlearned*, so a stale-but-still-ok
// agent choice can never be dethroned by a newly-better one without decay
// forcing it to compete again. The default (reward-modulated three-factor
// rule) can move weight down on failure and additionally decays every step.
export class HebbianRouter {
  constructor(agents, {
    eta = 0.2,
    decay = 0.02,
    epsilon = 0.1,
    temperature = 0.5,
    baselineRate = 0.05,
    meshStyle = false,
    rng,
  } = {}) {
    this.agents = agents;
    this.eta = eta;
    this.decay = meshStyle ? 0 : decay;
    this.epsilon = epsilon;
    this.temperature = temperature;
    this.baselineRate = baselineRate;
    this.meshStyle = meshStyle;
    this.rng = rng;
    this.w = new Map(); // feature -> Map(agent -> weight in [0, 1])
    this.baseline = 0.5; // running average reward (EMA)
    this.name = meshStyle ? 'hebbian-no-decay' : 'hebbian';
  }

  features(task) {
    return [task.type, ...task.keywords];
  }

  weightsFor(feature) {
    let m = this.w.get(feature);
    if (!m) {
      m = new Map(this.agents.map((a) => [a, 0]));
      this.w.set(feature, m);
    }
    return m;
  }

  scores(task) {
    const totals = Object.fromEntries(this.agents.map((a) => [a, 0]));
    for (const f of this.features(task)) {
      const wf = this.weightsFor(f);
      for (const a of this.agents) totals[a] += wf.get(a);
    }
    return totals;
  }

  select(task) {
    if (this.rng() < this.epsilon) {
      return this.agents[Math.floor(this.rng() * this.agents.length)];
    }
    const totals = this.scores(task);
    const probs = softmax(this.agents.map((a) => totals[a]), this.temperature);
    let r = this.rng();
    let cum = 0;
    for (let i = 0; i < this.agents.length; i++) {
      cum += probs[i];
      if (r <= cum) return this.agents[i];
    }
    return this.agents[this.agents.length - 1];
  }

  update(task, agent, reward) {
    const feats = this.features(task);
    const pre = 1 / feats.length; // normalized presynaptic activation
    // Reward-modulated ("three-factor") delta: can be negative on failure,
    // driving the chosen agent's weight down. meshStyle instead mirrors
    // mesh.rs's raw contrib1*contrib2*0.1 term: strengthen-only, no
    // baseline, so failure leaves the weight untouched rather than
    // punishing it.
    const modulated = this.meshStyle ? reward : reward - this.baseline;
    for (const f of feats) {
      const wf = this.weightsFor(f);
      for (const a of this.agents) {
        let w = wf.get(a) * (1 - this.decay);
        if (a === agent) w += this.eta * pre * modulated;
        wf.set(a, clamp(w, 0, 1));
      }
      // Normalisation: keep each feature row's total weight bounded so no
      // single feature can dominate every agent's score unboundedly.
      const total = [...wf.values()].reduce((s, x) => s + x, 0);
      const cap = this.agents.length * 0.5;
      if (total > cap) {
        const scale = cap / total;
        for (const a of this.agents) wf.set(a, wf.get(a) * scale);
      }
    }
    if (!this.meshStyle) this.baseline += this.baselineRate * (reward - this.baseline);
  }
}

// --- 4. PheromoneRouter ---------------------------------------------------
// ruflo's ADR-330 pheromone-adaptive topology; lives in its own module.
export { PheromoneRouter } from './pheromone.js';
