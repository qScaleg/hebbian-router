// BioMeshRouter: the three mechanisms of the neuralmesh-biological library
// (consolidation, neurogenesis, complementary memory) ported onto routing.
//
// Source (read-only, extracted newest version):
//   ~/.claude/jobs/435a339f/tmp/nmb-tgz/neuralmesh-biological/src/neuralmesh/
// Citations below are `file:line` relative to that `neuralmesh/` directory.
//
// The library trains a variational circuit by gradient descent on MSE
// (quantum/circuit.py:79-82) in a loop of probe -> grow? -> train with
// replay -> prune -> consolidate per task (trainer.py:1-11, 97-157). Here the
// "parameters" are routing affinities: each (taskType, agent) cell holds one
// parameter per layer, and its score is the sum over the row's layers.
//
// Mapping (ASSUMPTIONS are labelled where the library had nothing to copy):
//   - ASSUMPTION: one routed task = one optimisation step on a batch of one.
//     The loss is squared error between the chosen cell's score and the 0/1
//     reward (the library uses MSE against labels in {-1,+1}).
//   - ASSUMPTION: the library's task boundary (consolidate at the end of
//     fit_task, trainer.py:139) becomes every `consolidateEvery` routed
//     tasks. Default 40 = probe_steps 10 (growth.py:27) + steps_per_task 30
//     (trainer.py:52), i.e. one library "task" worth of steps.
//   - ASSUMPTION: selection. The library classifies and never selects, so we
//     use the same softmax + epsilon-greedy policy HebbianRouter uses.
//   - ASSUMPTION: importance. The library uses the diagonal quantum Fisher
//     (quantum/fisher.py:22-55), which has no routing analogue. We use the
//     classical empirical Fisher: the sum of squared task-loss gradients
//     per parameter since the last consolidation. From there the library's
//     own normalise / decay / freeze rules are applied unchanged.
import { softmax } from './routers.js';

// Library defaults, with the file:line each one comes from.
export const BIOMESH_DEFAULTS = Object.freeze({
  lr: 0.1, // trainer.py:51
  lam: 10, // ewc.py:31
  ewcDecay: 0.9, // ewc.py:33, importance = decay*importance + f (ewc.py:51)
  freezeThreshold: 0.85, // ewc.py:34, freeze if importance/max >= this (ewc.py:54-60)
  errorThreshold: 0.35, // growth.py:26, grow if probe loss > this (growth.py:40)
  probeSteps: 10, // growth.py:27, patience window before the growth check
  noveltyThreshold: 0.7, // growth.py:28, grow if frozen-importance fraction > this
  maxLayers: 8, // growth.py:29
  layersPerEvent: 1, // growth.py:55
  pruneActivityThreshold: 1e-3, // growth.py:56
  minGenerationAgeToPrune: 1, // growth.py:57
  activityEma: 0.9, // growth.py:78
  bufferCapacity: 128, // buffer.py:22
  perTaskCapacity: 64, // longterm.py:26
  sleepEvery: 10, // replay.py:29
  replayBatch: 16, // replay.py:30
  replayRatio: 0.5, // replay.py:31
  consolidateEvery: 40, // ASSUMPTION (see header)
  temperature: 0.05, // ASSUMPTION: HebbianRouter's tuned selection settings
  epsilon: 0.02, // ASSUMPTION: same
});

// --- complementary memory (memory/*.py) --------------------------------------

/** FIFO episodic buffer (buffer.py:19-44). */
export class EpisodicBuffer {
  constructor(capacity) {
    this.capacity = capacity;
    this.items = [];
  }

  push(exp) {
    this.items.push(exp);
    if (this.items.length > this.capacity) this.items.shift(); // deque(maxlen)
  }

  drain() {
    const out = this.items;
    this.items = [];
    return out;
  }

  get length() { return this.items.length; }
}

/** Sample min(n, arr.length) distinct items without replacement. */
function sampleWithoutReplacement(rng, arr, n) {
  const idx = arr.map((_, i) => i);
  const k = Math.min(n, arr.length);
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rng() * (idx.length - i));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, k).map((i) => arr[i]);
}

/** Task-balanced reservoir store (longterm.py:12-68). Each "task" gets its own
 * Algorithm-R reservoir (longterm.py:31-46), and sampling goes round-robin
 * over tasks in insertion order (longterm.py:48-62). */
export class LongTermStore {
  constructor(perTaskCapacity) {
    this.perTaskCapacity = perTaskCapacity;
    this.store = new Map();
    this.seen = new Map();
  }

  integrate(items, rng) {
    for (const e of items) {
      const seen = (this.seen.get(e.task) ?? 0) + 1;
      this.seen.set(e.task, seen);
      if (!this.store.has(e.task)) this.store.set(e.task, []);
      const res = this.store.get(e.task);
      if (res.length < this.perTaskCapacity) res.push(e);
      else {
        const j = Math.floor(rng() * seen);
        if (j < this.perTaskCapacity) res[j] = e;
      }
    }
  }

  sample(n, rng, excludeTask) {
    const tasks = [...this.store.keys()].filter((t) => t !== excludeTask && this.store.get(t).length);
    if (!tasks.length) return [];
    const per = Math.max(1, Math.floor(n / tasks.length));
    const out = [];
    for (const t of tasks) out.push(...sampleWithoutReplacement(rng, this.store.get(t), per));
    return out.slice(0, n); // longterm.py:62, so with many tasks the OLDEST win
  }

  get length() {
    let s = 0;
    for (const v of this.store.values()) s += v.length;
    return s;
  }
}

/** Sleep cycle (replay.py:45-63): drain the buffer into the long-term store,
 * then build a batch that is replayRatio old (other tasks) and the rest
 * recent. Returns { batch, recentCount } so tests can check the bounds. */
export function sleepCycle(buffer, store, rng, { replayBatch, replayRatio }, currentTask) {
  const recent = buffer.drain();
  store.integrate(recent, rng);
  const nOld = Math.floor(replayBatch * replayRatio);
  let nNew = replayBatch - nOld;
  const old = store.sample(nOld, rng, currentTask);
  if (!old.length) nNew = replayBatch; // replay.py:55-56
  const fresh = sampleWithoutReplacement(rng, recent, nNew);
  return { batch: [...old, ...fresh], recentCount: recent.length };
}

// --- the router ----------------------------------------------------------------

export class BioMeshRouter {
  constructor(agents, taskTypes, {
    rng,
    name = 'biomesh',
    consolidation = true,
    replay = true,
    neurogenesis = true,
    ...opts
  } = {}) {
    this.agents = agents;
    this.rng = rng;
    this.name = name;
    this.useEwc = consolidation;
    this.useReplay = replay;
    this.useGrowth = neurogenesis;
    this.cfg = { ...BIOMESH_DEFAULTS, ...opts };
    this.params = []; // flat list of every parameter across all layers
    this.rows = new Map();
    for (const t of taskTypes) {
      const row = { type: t, layers: [], generation: 0, segObs: 0, probeLoss: 0, checked: false, growths: 0 };
      this.rows.set(t, row);
      this.addLayer(row, 0);
    }
    this.buffer = new EpisodicBuffer(this.cfg.bufferCapacity);
    this.store = new LongTermStore(this.cfg.perTaskCapacity);
    this.globalStep = 0;
    this.routed = 0;
    this.segment = 0; // stands in for the library's task name
    this.log = []; // growth / prune / consolidate events
  }

  // growth.py:59-76 plus mesh.py:104-129: append a layer tagged with a new
  // generation, leaving old parameters untouched. mesh.py:119 initialises
  // new angles uniform(-pi, pi) * 0.1. ASSUMPTION: a routing affinity starts
  // at 0 instead, so a new layer only adds capacity, not noise.
  addLayer(row, generation) {
    const layer = Object.fromEntries(this.agents.map((a) => {
      const p = { type: row.type, agent: a, layer: row.layers.length, generation,
        value: 0, anchor: 0, importance: 0, frozen: false, activity: 0, fisher: 0 };
      this.params.push(p);
      return [a, p];
    }));
    row.layers.push(layer);
  }

  score(type, agent) {
    let s = 0;
    for (const layer of this.rows.get(type).layers) s += layer[agent].value;
    return s;
  }

  select(task) {
    if (this.rng() < this.cfg.epsilon) return this.agents[Math.floor(this.rng() * this.agents.length)];
    const probs = softmax(this.agents.map((a) => this.score(task.type, a)), this.cfg.temperature);
    const r = this.rng();
    let cum = 0;
    for (let i = 0; i < this.agents.length; i++) {
      cum += probs[i];
      if (r <= cum) return this.agents[i];
    }
    return this.agents[this.agents.length - 1];
  }

  // trainer.py:79-88 (_step): gradient of mean-batch MSE plus the EWC
  // penalty 0.5*lam*sum F*(theta - anchor)^2 (ewc.py:64-68), whose gradient
  // is lam*F*(theta - anchor). Frozen parameters get zero gradient
  // (ewc.py:70-75). Activity is recorded (growth.py:78-83), then theta -= lr*grad.
  step(items, { accumulateFisher }) {
    const grad = new Map();
    for (const it of items) {
      const err = this.score(it.type, it.agent) - it.reward;
      for (const layer of this.rows.get(it.type).layers) {
        const p = layer[it.agent];
        grad.set(p, (grad.get(p) ?? 0) + (2 * err) / items.length);
      }
    }
    if (accumulateFisher) for (const [p, g] of grad) p.fisher += g * g;
    const { lam, lr, activityEma } = this.cfg;
    for (const p of this.params) {
      let g = grad.get(p) ?? 0;
      if (this.useEwc) g += lam * p.importance * (p.value - p.anchor);
      if (p.frozen) g = 0;
      p.activity = activityEma * p.activity + (1 - activityEma) * Math.abs(g);
      p.value -= lr * g;
    }
    this.globalStep += 1;
  }

  update(task, agent, reward) {
    const row = this.rows.get(task.type);
    this.step([{ type: task.type, agent, reward }], { accumulateFisher: true });
    if (this.useReplay) this.buffer.push({ type: task.type, agent, reward, task: this.segment, step: this.globalStep });

    // Neurogenesis check (trainer.py:100-116, growth.py:31-48), per row. The
    // library probes for probe_steps and then decides once per task.
    // ASSUMPTION: the probe loss is the mean post-update squared error over
    // the row's first probeSteps observations in the segment, averaging in
    // place of the library's batch of 8. The EWC term the library includes
    // in the probe loss (trainer.py:104) is omitted.
    row.segObs += 1;
    if (row.segObs <= this.cfg.probeSteps) {
      const e = this.score(task.type, agent) - reward;
      row.probeLoss += (e * e) / this.cfg.probeSteps;
    }
    if (this.useGrowth && !row.checked && row.segObs === this.cfg.probeSteps) {
      row.checked = true;
      this.maybeGrow(row);
    }

    // Sleep (trainer.py:127-132): one extra EWC-penalised step on the batch.
    if (this.useReplay && this.globalStep > 0 && this.globalStep % this.cfg.sleepEvery === 0) {
      const { batch } = sleepCycle(this.buffer, this.store, this.rng, this.cfg, this.segment);
      if (batch.length) this.step(batch, { accumulateFisher: false });
    }

    this.routed += 1;
    if (this.routed % this.cfg.consolidateEvery === 0) this.endSegment();
  }

  maybeGrow(row) {
    // Novelty input: the share of importance sitting on frozen parameters
    // (trainer.py:107-109). ASSUMPTION: measured per row, not over the
    // whole mesh, because growth is per row here.
    let imp = 0;
    let frozenImp = 0;
    for (const layer of row.layers) {
      for (const p of Object.values(layer)) {
        imp += p.importance;
        if (p.frozen) frozenImp += p.importance;
      }
    }
    const frozenFrac = imp > 0 ? frozenImp / imp : 0;
    const { maxLayers, errorThreshold, noveltyThreshold, layersPerEvent } = this.cfg;
    let reason = null;
    if (row.layers.length >= maxLayers) reason = null; // growth.py:38-39
    else if (row.probeLoss > errorThreshold) reason = 'probe_loss'; // growth.py:40-41
    else if (frozenFrac > noveltyThreshold) reason = 'frozen_importance_fraction'; // growth.py:42-47
    if (!reason) return false;
    row.generation += 1;
    for (let i = 0; i < layersPerEvent && row.layers.length < maxLayers; i++) this.addLayer(row, row.generation);
    row.growths += 1;
    this.log.push({ event: 'grow', type: row.type, reason, routed: this.routed, segObs: row.segObs });
    return true;
  }

  endSegment() {
    // 4. prune (trainer.py:136, growth.py:85-102): freeze parameters whose
    // activity stays in (0, threshold), in layers at least one generation
    // old. ASSUMPTION: generations are per row, since growth is per row.
    if (this.useGrowth) {
      for (const row of this.rows.values()) {
        for (const layer of row.layers) {
          for (const p of Object.values(layer)) {
            const oldEnough = row.generation - p.generation >= this.cfg.minGenerationAgeToPrune;
            if (oldEnough && p.activity < this.cfg.pruneActivityThreshold && p.activity > 0 && !p.frozen) {
              p.frozen = true;
              this.log.push({ event: 'prune', type: p.type, agent: p.agent, layer: p.layer });
            }
          }
        }
      }
    }
    // 5. consolidate (trainer.py:139, ewc.py:43-61): clip, normalise by max,
    // importance = decay*importance + f, snapshot anchors, and hard-freeze
    // every parameter whose importance/max >= freezeThreshold.
    if (this.useEwc) {
      const maxF = Math.max(0, ...this.params.map((p) => p.fisher));
      for (const p of this.params) {
        const f = maxF > 0 ? Math.max(0, p.fisher) / maxF : 0;
        p.importance = this.cfg.ewcDecay * p.importance + f;
        p.anchor = p.value;
      }
      const maxI = Math.max(0, ...this.params.map((p) => p.importance));
      if (maxI > 0 && this.cfg.freezeThreshold != null) {
        for (const p of this.params) if (p.importance / maxI >= this.cfg.freezeThreshold) p.frozen = true;
      }
    }
    for (const p of this.params) p.fisher = 0;
    // trainer.py:141: the finished task goes into long-term memory.
    if (this.useReplay) this.store.integrate(this.buffer.drain(), this.rng);
    for (const row of this.rows.values()) { row.segObs = 0; row.probeLoss = 0; row.checked = false; }
    this.segment += 1;
  }

  frozenCount() {
    return this.params.filter((p) => p.frozen).length;
  }
}
