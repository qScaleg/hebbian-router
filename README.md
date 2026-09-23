# Hebbian router experiment

Does a biologically-inspired Hebbian "agent-affinity" router beat simpler
task routers? This is a small, self-contained simulation to find out, built
because two real systems suggested the question:

- **Synaptic-Mesh** (`src/rs/neural-mesh/src/mesh.rs`, lines ~92-105)
  strengthens a connection by `contrib1 * contrib2 * 0.1`, capped at `1.0`,
  with **no decay** and **no negative correction on failure**.
- **ruflo**'s router uses static keyword/semantic matching for agent
  selection and a Thompson-sampling bandit for model choice.

This experiment builds a synthetic world of tasks and agents with hidden
success probabilities, and compares several routers (including a port of
ruflo's pheromone-adaptive topology) on how well they assign
tasks to agents, purely from observed outcomes (reward = success/failure).

## What's in here

- `src/prng.js` — seeded `mulberry32` PRNG plus small helpers (shuffle,
  Bernoulli, Beta sampling via the Gamma-ratio method).
- `src/simulator.js` — task types, agents, a hidden success-probability
  table (with two *non-obvious* pairings a keyword router gets wrong), noisy
  keyword generation, and a mid-run skill drift.
- `src/routers.js` — `KeywordRouter` (static map, no learning),
  `ThompsonRouter` (Beta posterior per task-type x agent), and
  `HebbianRouter`, a reward-modulated ("three-factor") Hebbian rule with
  weight decay and normalisation, plus a `meshStyle` option that mirrors
  mesh.rs's literal update (strengthen-only, no decay, no negative
  correction).
- `src/pheromone.js` — a port of ruflo's `pheromone-adaptive` topology
  (ADR-330) plus `PheromoneRouter`, which runs it in this simulation (see
  "ruflo pheromone-adaptive" below).
- `src/experiment.js` — runs N seeds x T tasks, prints a results table and
  paired per-seed comparisons against `hebbian`, and writes
  `results/results.json`.
- `src/grid-pheromone.js` — the pheromone hyperparameter grid search.
- `src/biomesh.js` — `BioMeshRouter`, the neuralmesh-biological
  mechanisms (EWC consolidation, neurogenesis, complementary replay memory)
  ported onto routing, plus `src/grid-biomesh.js`, its grid search.
- `test/*.test.js` — unit tests for the PRNG, the world, and each router,
  plus one integration test of the full experiment pipeline.

## Running it

```bash
node --test                  # unit + integration tests (42 tests)
node src/experiment.js       # runs the experiment, prints the table, writes results/results.json
node src/grid-pheromone.js   # reruns the pheromone grid search (~2 s)
node src/grid-biomesh.js     # reruns the biomesh grid search (~4 s)
```

No npm dependencies — plain ESM, Node's built-in `node:test` runner, Node 24.

## The world

Six task types (`security`, `api`, `testing`, `docs`, `perf`, `refactor`)
and six agents (`coder`, `tester`, `reviewer`, `security-auditor`,
`architect`, `docs-writer`). Each task carries noisy keyword text (2-3
keywords for its type, plus a 25%-chance confound keyword borrowed from a
different type) that routers must work from.

Four task types have an "obvious" best agent that also matches the static
keyword map (e.g. `security` keywords -> `security-auditor`, who really is
best). Two are deliberately **non-obvious traps**: `perf` keywords map to
`coder` in the static keyword table, but the hidden best agent is actually
`architect`; `refactor` keywords map to `coder`, but the hidden best agent
is `reviewer`. A pure keyword router can never discover this; an
outcome-learning router can.

Halfway through each run (task 1000 of 2000), skills **drift**: for `perf`,
`architect`'s edge fades (0.75 -> 0.55) while `coder` gets much better
(0.45 -> 0.75); for `refactor`, `reviewer` fades (0.78 -> 0.55) while
`architect` gets much better (0.45 -> 0.82). The decline is deliberately
*mild*, not catastrophic — the old favorite still succeeds a majority of
the time. This matters: if the old favorite instead started failing
outright, any router that punishes failure would "accidentally" recover
fast regardless of whether it has real decay. A mild decline isolates what
decay is actually for — forgetting a stale-but-still-okay association fast
enough for a newly-better option to win.

## The routers

- **KeywordRouter** — a fixed keyword -> agent lookup table (like a simple
  default routing config). No memory, no learning, deterministic.
- **ThompsonRouter** — a Beta-Bernoulli bandit keyed by (task type, agent),
  sampling from posteriors and updating alpha/beta on outcomes.
- **HebbianRouter** — weights `w[feature][agent]` over features (the task's
  type plus its keywords). Selection is softmax-over-summed-weights with
  epsilon-greedy exploration. Update rule per task:

  ```
  delta = eta * pre * (reward - baseline)   // only for the chosen agent
  w = clamp(w * (1 - decay) + delta, 0, 1)  // decay, then the update, then cap at 1.0
  ```

  `baseline` is a running EMA of reward (variance reduction / the "third
  factor" in reward-modulated Hebbian learning). Each feature's weight row
  is rescaled if its total exceeds a cap, so no single feature can
  dominate every agent's score unboundedly.

  Passing `meshStyle: true` reproduces mesh.rs's actual rule instead:
  `delta = eta * pre * reward` (no baseline subtraction, so failure applies
  **no** update at all — not even a small one), decay forced to `0`. This
  variant's weights can only grow or hold, never shrink except through
  normalisation.

### Hyperparameter selection

`HebbianRouter`'s `eta`/`epsilon`/`temperature` were picked with a small
grid search (32-60 configurations x 8 seeds x 2000 tasks) rather than left
at arbitrary defaults, to give it a fair comparison instead of a strawman
one. Best found: `eta=0.5, epsilon=0.02, temperature=0.05, decay=0.01`
(see `HEBBIAN_OPTS` in `src/experiment.js`). Sloppier settings (higher
temperature especially) tanked performance to worse than random — softmax
temperature matters a lot here, more than decay or epsilon in isolation.

## ruflo pheromone-adaptive (ADR-330)

`src/pheromone.js` ports ruflo's actual implementation rather than the
idea of it. Sources: `ruflo/v3/docs/adr/ADR-330-adaptive-pheromone-swarm-consensus.md`
and `ruflo/v3/@claude-flow/cli/src/services/pheromone-adaptive.ts`, checked
against the installed build
`<npm root -g>/ruflo/node_modules/@claude-flow/cli/dist/src/services/pheromone-adaptive.js`
(ruflo CLI 3.42.5). `recordApscSignal` is a line-for-line port. It mutates
state in place instead of copying it, and uses the round number instead of a
wall-clock timestamp, which gives the same "stalest first" ordering.

### The rules as ruflo implements them (`DEFAULT_APSC_CONFIG`)

| Rule | Value / formula |
|---|---|
| Fitness | `raw = α·taskSuccess + β·(1 − normalizedLatency) + γ·consensusAlignment`, α=0.5, β=0.2, γ=0.3, renormalised to sum 1, inputs must be in [0,1] |
| Role normalisation | `normalized = clamp(0.5 + raw − roleBaseline, 0, 1)`. `roleBaseline` is the mean latest `raw` of *other* agents in the same role, else the role's EMA baseline, else `raw` itself |
| Agent EMA | `ema = 0.85·prevEma + 0.15·normalized` (`emaDecay` 0.85; the first observation sets `ema = normalized`) |
| Role baseline EMA | `0.85·priorRoleBaseline + 0.15·mean(latest raw of every agent in the role)` |
| Adaptive threshold | starts at 0.5; `0.85·threshold + 0.15·mean(ema of agents with ≥ minSamples)` |
| Pause (suspend) | active, not protected, `samples ≥ minSamples (3)`, and `ema < threshold × pruningFactor (0.6)` |
| Reactivate | a suspended agent's own update with `ema ≥ threshold × reactivationThreshold (0.75)`, or exploration: every `round(1/explorationRate)` = 10 rounds, the stalest suspended agent is reactivated |
| Minimum-active floor | `minActiveAgents = 3`: no suspension if `active − 3 < 1` |
| Max paused per round | `maxSuspendFraction = 0.25`: no suspension if `floor(active × 0.25) < 1` |
| Protected roles | `coordinator`, `queen`, `security-architect`, `security-auditor` |
| Mode | `dryRun: true` by default (records `would-suspend`, never denies); live only with `--apsc-live` |

What it is *not*: APSC does not select agents. It is a scheduling
**eligibility gate**. `agent_execute` (`mcp-tools/agent-tools.js`) refuses
an agent only when a live APSC swarm has suspended it. Its state is one per
swarm, keyed by agent ID, and has no notion of task type. `hooks_post-task`
(`mcp-tools/hooks-tools.js`) feeds it: `taskSuccess = success ? 1 : 0`,
`normalizedLatency = duration / latencyBudgetMs`,
`consensusAlignment = consensusAlignment ?? quality`, and
`role = agentRole ?? agentId`.

### Mapping it onto this simulation (assumptions labelled)

- `taskSuccess` = the task's 0/1 reward.
- `normalizedLatency` = **ASSUMPTION: neutral 0.5 constant**. The
  simulator has no latency.
- `consensusAlignment` = **ASSUMPTION: neutral 0.5 constant**. There is
  no consensus step. Both constants add the same amount to every raw score,
  so role centring cancels them. Only success moves fitness.
- Role: `roleMode: 'per-agent'` (role = agent name) is what
  `hooks_post-task` does by default. `'shared'` (every agent in one role)
  is a legal ruflo config but not the default.
- **`pheromone` (faithful, `mode: 'gate'`)**: `KeywordRouter` picks, and
  live APSC vetoes suspended picks, with one swarm-wide state.
  **ASSUMPTION:** ruflo just returns an error for a suspended pick; here
  the task is re-routed to the eligible agent with the highest EMA.
  (Dry-run, ruflo's default, routes exactly like `keyword`; a test checks
  this.)
- **`pheromone-select` (extension, NOT ruflo behaviour)**: routes directly
  on pheromone fitness. **ASSUMPTIONS:** one APSC state per task type,
  because a context-free per-agent score cannot learn "architect for perf";
  agents below `minSamples` are tried first; otherwise the argmax EMA among
  eligible agents, with a random tie-break. There is no exploration beyond
  APSC's own reactivation cadence.

### Pheromone grid search (disclosed)

`src/grid-pheromone.js` uses the same budget as the Hebbian search: 48
configurations per mode x 8 seeds x 2000 tasks, drift at 1000. The grid
covers `emaDecay` {0.5, 0.7, 0.85, 0.95}, `pruningFactor` {0.4, 0.6, 0.8},
`explorationRate` {0.05, 0.1} and `roleMode` {per-agent, shared}. The
weights, `reactivationThreshold` and `minSamples` stay at ruflo defaults.
The floor and max-fraction settings are safety invariants, not knobs. Tuning
used seeds 1-8, the first 8 of the 30 evaluation seeds. The Hebbian search's
seeds were not recorded, so this is the choice at least as generous to
pheromone. Findings:

- **gate**: the top configs all tie at 0.7044, which is the keyword
  router's own score on those seeds, and none of them ever suspends an
  agent. The worst config (shared role, `pruningFactor` 0.8) drops to 0.54:
  when the gate fires at all, it hurts. The tuned row uses the first tied
  config.
- **select**: `roleMode: 'shared'` plus `emaDecay` 0.7 is best (0.7228 on
  the tuning seeds). `pruningFactor` and `explorationRate` made no
  difference. Every per-agent-role config sits near 0.23-0.26.

## neuralmesh-biological mechanisms (BioMeshRouter)

`src/biomesh.js` ports the three mechanisms of the neuralmesh-biological
library (`src/neuralmesh/`, read-only copy of the newest version). Each rule
cites `file:line` in the code. The parameters are the (taskType, agent)
affinity table: each cell holds one parameter per layer, and its score is
the sum over its row's layers.

| Mechanism | Library rule (source) | Routing port |
|---|---|---|
| Learning step | MSE gradient step, lr 0.1 (`trainer.py:79-88`, `quantum/circuit.py:79-82`) | squared error of cell score vs 0/1 reward, lr 0.1 |
| EWC penalty | `0.5·λ·Σ F·(θ−θ*)²`, λ=10 (`ewc.py:31,64-68`) | same, applied to every parameter every step |
| Importance | QFIM diagonal, clipped, / max, `I = 0.9·I + f` (`ewc.py:45-51`) | **ASSUMPTION:** empirical Fisher (Σ squared task gradients since last consolidation) in place of the QFIM; the normalise/decay steps are the library's |
| Hard freeze | `I/max(I) ≥ 0.85` → frozen, gradient zeroed, permanent (`ewc.py:54-60,70-75`) | same |
| Consolidation point | end of each task (`trainer.py:139`), 10 probe + 30 train steps per task | **ASSUMPTION:** every 40 routed tasks |
| Growth trigger | probe loss > 0.35 after 10 probe steps, or frozen-importance fraction > 0.7; max 8 layers (`growth.py:26-48`, `trainer.py:100-116`) | per row, once per segment after the row's 10th observation. **ASSUMPTION:** probe loss = mean squared error over those 10, and the fraction is measured per row |
| Growth action | append a new-generation layer, old nodes untouched (`growth.py:59-76`) | append a fresh layer to that row. **ASSUMPTION:** initialised at 0 (the library uses `uniform(−π,π)·0.1`, `mesh.py:119`) |
| Pruning | freeze params whose activity EMA(0.9) of \|grad\| is in (0, 1e-3), at least 1 generation old (`growth.py:56-57,78-102`) | same, per parameter, generations per row |
| Episodic buffer | FIFO, capacity 128 (`buffer.py:19-44`) | same, holding (type, agent, reward, segment) |
| Long-term store | Algorithm-R reservoir per task, 64 each; round-robin sampling truncated to n (`longterm.py:26-62`) | same. **ASSUMPTION:** a library "task" is a consolidation segment (a phase in time), so old = earlier segments |
| Sleep | every 10 steps: drain buffer → store, batch 16 = 50% old + 50% recent, one EWC-penalised step (`replay.py:29-63`, `trainer.py:127-132`) | same |
| Selection | none (the library classifies) | **ASSUMPTION:** HebbianRouter's softmax + epsilon-greedy |

Ablations: `biomesh-sgd` (none of the three; this is the library's
"plain fine-tuning" baseline), `biomesh-consolidation-only` (consolidation only),
`biomesh-consolidation+replay`, `biomesh-full` (all three), and
`biomesh-full-defaults` (all three at the library's own defaults, no tuning).

**Grid search (disclosed).** `src/grid-biomesh.js` tunes each variant
separately, on seeds 1-8 like the others, over `lr` {0.1, 0.3},
`temperature` {0.05, 0.1, 0.2}, `λ` {1, 10, 50} and `consolidateEvery`
{40, 200}. That is 36 configs for the EWC variants and 6 for SGD, within
the Hebbian budget. Every EWC variant's best config is the weakest
consolidation on offer (λ=1, consolidating every 200). The worst configs
(λ=50) fall to 0.34-0.43. The tuner's answer is "as little consolidation
as possible".

Diagnostics (30 seeds, tuned configs):
- `biomesh-consolidation-only`: ends with about 4 of 36 cells hard-frozen.
- `biomesh-full`: about 9 growth events per run. 8.7 of them come from the
  frozen-importance "novelty" trigger and 0.3 from the probe-loss trigger;
  Bernoulli rewards rarely push MSE above 0.35. Only about 3 of the 9 happen
  after the drift. About 18 parameters are pruned, and the table grows from
  36 to about 90 parameters.
- At library defaults: about 16 growths, 45 prunes and 51 frozen parameters.

## Results

30 seeds x 2000 tasks/seed, drift at task 1000. From an actual run
(`results/results.json` has the full data):

```
router                        successRate  totalRegret  preDrift  earlyPostDrift  latePostDrift
----------------------------  -----------  -----------  --------  --------------  -------------
keyword                       0.7011       210.2057     0.6774    0.7107          0.6947
thompson                      0.7234       164.3503     0.7126    0.7007          0.7333
hebbian                       0.7239       166.9473     0.7027    0.6860          0.7493
hebbian-no-decay              0.5180       574.7660     0.5142    0.4907          0.5087
pheromone                     0.7011       210.2057     0.6774    0.7107          0.6947
pheromone-select              0.2534       1100.7607    0.2965    0.2260          0.1833
pheromone-tuned               0.7011       210.2057     0.6774    0.7107          0.6947
pheromone-select-tuned        0.7044       202.5973     0.6910    0.6813          0.7067
biomesh-sgd                   0.7226       163.5750     0.7025    0.6807          0.7367
biomesh-consolidation-only    0.6885       232.5210     0.6768    0.6740          0.6853
biomesh-consolidation+replay  0.6662       278.0183     0.6675    0.6453          0.6473
biomesh-full                  0.7025       207.7677     0.7002    0.6633          0.6873
biomesh-full-defaults         0.5603       491.2000     0.5740    0.5440          0.4993
```

Paired per-seed differences (hebbian minus other, same task and outcome
streams; for regret, + means hebbian had less). Values are mean ± 95% t-CI,
with seed wins shown as hebbian-other:

```
keyword / pheromone / pheromone-tuned  success +0.0228 ± 0.0087 (28-2)  regret +43.3 ± 16.5 (28-2)
thompson                               success +0.0005 ± 0.0072 (16-14) regret  -2.6 ± 15.0 (15-15)
pheromone-select                       success +0.4705 ± 0.0108 (30-0)  regret +933.8 ± 18.4 (30-0)
pheromone-select-tuned                 success +0.0195 ± 0.0163 (18-12) regret +35.7 ± 33.5 (18-12)
biomesh-sgd                            success +0.0013 ± 0.0107 (14-16) regret  -3.4 ± 21.4 (12-18)
biomesh-consolidation-only             success +0.0354 ± 0.0146 (24-6)  regret +65.6 ± 29.3 (23-7)
biomesh-consolidation+replay           success +0.0577 ± 0.0192 (24-6)  regret +111.1 ± 39.2 (24-6)
biomesh-full                           success +0.0214 ± 0.0142 (21-9)  regret +40.8 ± 29.5 (22-8)
biomesh-full-defaults                  success +0.1637 ± 0.0258 (30-0)  regret +324.3 ± 49.3 (30-0)
```

Each biomesh mechanism against plain SGD on the same table (variant minus
SGD, so + means the mechanism helped; mean ± 95% CI over 30 paired seeds):

```
                              overall          preDrift         earlyPostDrift   latePostDrift
biomesh-consolidation-only    -0.034 ± 0.013   -0.026 ± 0.024   -0.007 ± 0.022   -0.051 ± 0.024
biomesh-consolidation+replay  -0.056 ± 0.019   -0.035 ± 0.026   -0.035 ± 0.026   -0.089 ± 0.035
biomesh-full                  -0.020 ± 0.013   -0.002 ± 0.020   -0.017 ± 0.018   -0.049 ± 0.023
biomesh-full-defaults         -0.162 ± 0.027   -0.128 ± 0.029   -0.137 ± 0.034   -0.237 ± 0.041
```

`biomesh-full` against the two learning baselines (biomesh-full minus
other, + means biomesh-full is better; seed wins are full-other):

```
              overall                 preDrift                earlyPostDrift          latePostDrift
vs hebbian    -0.021 ± 0.014 (9-21)   -0.002 ± 0.020 (17-13)  -0.023 ± 0.020 (7-15)   -0.062 ± 0.023 (1-23)
vs thompson   -0.021 ± 0.011 (7-23)   -0.012 ± 0.015 (13-16)  -0.037 ± 0.019 (6-19)   -0.046 ± 0.022 (4-23)
```

Adding routers does not change existing rows: each router owns its RNG,
and the task and outcome streams are shared.

For context: a uniformly-random agent choice averages **0.392** success
across all (type, agent) pairs pre-drift; an oracle that always picks the
true-best agent averages **0.800**.

`earlyPostDrift`/`latePostDrift` are the mean reward in the first/last 50
tasks after the drift point (task 1000-1050 and 1950-2000).

## Conclusions (honest)

**The tuned reward-modulated Hebbian router does not clearly beat Thompson
sampling.** It essentially ties it (0.7239 vs 0.7234 success rate, and
slightly *worse* total regret: 166.9 vs 164.3). Given the overlapping
confidence one would get from 30 seeds, this is a wash, not a win. If
anything Hebbian pulls slightly ahead only after the world fully
re-stabilizes late post-drift (0.7493 vs 0.7333), suggesting its
continuously-decaying weights let it settle into a *slightly* sharper
final policy once evidence has accumulated — but the margin is small and
not worth over-interpreting from this run count.

**The keyword router is a surprisingly strong baseline** (0.7011) because
four of six task types have no trap at all — a static lookup is exactly
right for `security`, `api`, `testing`, and `docs`. Its only losses come
from the `perf`/`refactor` traps, and by coincidence the `perf` drift
*helps* it (post-drift, `coder` really does become the right call for
`perf`, which the keyword map already assumed), partially masking its
blind spot. This is a caution about synthetic benchmarks in general: a
"dumb" router looks much better whenever most of the world is easy, and
the interesting comparisons only show up in the traps.

**The literal mesh.rs-style update rule is bad — clearly, not
subtly.** `hebbian-no-decay` (strengthen-only, no decay, no negative
correction) scores 0.518, barely above the 0.392 random-choice floor and
far below every other router, both before *and* after drift. This isn't
only a drift-adaptation problem: because weights can never be pushed back
down when a mediocre agent gets lucky, and averaging/normalisation ends up
diluting the whole feature row across every agent that ever succeeded once,
the router loses its ability to discriminate between agents at all, even
in the stationary first half of the run. This matches the concern raised
about `mesh.rs`: a Hebbian rule with no decay and no failure signal is not
a minor simplification, it is a materially worse learning rule.

**Takeaway:** Hebbian *can* be made competitive with a well-tuned Thompson
sampler, but only with real hyperparameter tuning and the reward-modulated
three-factor form (decay + baseline-subtracted updates). The naive
biologically-literal version (like the one in `mesh.rs`) underperforms
badly. If you're choosing a router for a task-routing setting like this,
Thompson sampling remains the simpler, equally-good, and less
tuning-sensitive choice — Hebbian's only edge here is a slightly sharper
late-stationary-state policy, which is a thin justification for its added
complexity and tuning burden.

### Hebbian vs ruflo pheromone-adaptive

**Yes, Hebbian adds something over ruflo's pheromone-adaptive routing in
this world, but only because pheromone-adaptive is not a learning router
in the first place.** As shipped, APSC is an eligibility gate on top of
whatever router picks the agent. With ruflo's defaults it never suspended a
single agent in 30 x 2000 tasks. So `pheromone` is bit-for-bit the keyword
router, and Hebbian beats it by +0.023 success (95% CI ±0.009, 28 of 30
seeds) and 43 less regret: strong evidence, but it is the known
keyword-vs-learner gap. No gate config in the grid did better. The ones that
fired only removed decent agents and scored worse.

Turning APSC into a selector (`pheromone-select`, an extension ruflo does
not have) fails completely with the default per-agent roles: 0.25, below
random. Centring each agent on its own baseline turns fitness into
"surprise versus its own past". A weak agent's rare success
(`0.5 + 0.5·(1 − p)`) lifts its EMA more than a strong agent's routine
success does, so the greedy choice drifts to weak agents. With a shared role
and tuned `emaDecay`, it reaches 0.704, and Hebbian still leads by
+0.020 ± 0.016 (18-12 seeds, 36 less regret ± 33). That margin is
borderline: the CI only just excludes zero, and it widens because the tuned
selector locks onto a wrong agent on some seeds. Overall, the only place
Hebbian clearly beats pheromone is where pheromone behaves like the keyword
router. Hebbian is still not distinguishable from plain Thompson sampling
(+0.0005 ± 0.007, 16-14). If ruflo wanted outcome-learned routing, a
per-task-type Thompson bandit is the simpler addition. APSC's
suspension/safety layer could sit on top of any of these routers,
because it solves a different problem: quarantining failing agents safely,
not choosing the best one.

### neuralmesh-biological mechanisms (BioMeshRouter)

**No. None of the three mechanisms helps routing here. Consolidation costs
plasticity after drift, and neurogenesis only partly buys it back.** The
plain-SGD version of the same table (`biomesh-sgd`, 0.7226) ties Hebbian
(−0.001 ± 0.011) and Thompson. Every mechanism added makes it worse, with
the size of the hit depending on the mechanism:

- **Consolidation (EWC plus the freeze)** costs −0.034 ± 0.013 overall. The
  cost is concentrated late after drift (−0.051 ± 0.024). Right after the
  drift the difference is not significant (−0.007 ± 0.022). Anchors and
  frozen cells hold the stale pre-drift affinities, so the router
  re-converges to a worse policy rather than reacting more slowly. This is
  the stability–plasticity cost, and it is clear over 30 seeds.
- **Replay makes it worse** (−0.056 ± 0.019; late post-drift −0.089). The
  long-term store is balanced across past segments, and its round-robin
  sample truncated to n (`longterm.py:62`) favours the *oldest* segments.
  So after the drift, half of every sleep batch rehearses pre-drift
  outcomes: protection against forgetting, applied to a world where
  forgetting is the goal.
- **Neurogenesis partly recovers it** (full −0.020 ± 0.013, and pre-drift
  back to parity: −0.002 ± 0.020). New layers give frozen rows fresh
  capacity. But growth fires on frozen-importance accumulation, not on the
  drift (probe loss almost never exceeds 0.35 with Bernoulli rewards), so
  late post-drift is still −0.049 ± 0.023.
- **Against the learning baselines**, `biomesh-full` loses to Hebbian
  (−0.021 ± 0.014, 9-21 seeds) and to Thompson (−0.021 ± 0.011, 7-23). It
  is at parity only before the drift and loses late after it: −0.062
  (1-23 seeds) vs Hebbian and −0.046 (4-23) vs Thompson.
- **Library defaults** (λ=10, consolidating every 40) are much worse:
  0.560, −0.16 vs SGD, with 30/30 seeds losing to Hebbian.

This matches the library's own 5-seed sweep, where the mesh also fails to
beat plain fine-tuning (forgetting 0.125-0.41 vs 0.094-0.125). Evidence
strength: the overall and late-post-drift deficits have 95% CIs that
exclude zero over 30 paired seeds. The early-post-drift differences for
`biomesh-consolidation-only` and `biomesh-full` do not. The grid search choosing the
weakest consolidation available for every variant says the same thing
independently. For a routing table under drift, the useful piece of this
design is its plain gradient step; the consolidation machinery works
against it.
