// Deterministic PRNG utilities (no external deps).
// mulberry32 is a small, fast, well-distributed 32-bit PRNG.

/**
 * Create a seeded PRNG function returning floats in [0, 1).
 * Same seed -> same sequence, forever (used for reproducible experiments).
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random integer in [0, maxExclusive). */
export function randInt(rng, maxExclusive) {
  return Math.floor(rng() * maxExclusive);
}

/** Pick a random element from an array. */
export function choice(rng, arr) {
  return arr[randInt(rng, arr.length)];
}

/** Bernoulli trial: 1 with probability p, else 0. */
export function bernoulli(rng, p) {
  return rng() < p ? 1 : 0;
}

/** Fisher-Yates shuffle, returns a new array (does not mutate input). */
export function shuffle(rng, arr) {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = randInt(rng, i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** Exponential(1) sample via inverse-CDF; guards against log(0). */
export function sampleExponential(rng) {
  const u = 1 - rng(); // u in (0, 1]
  return -Math.log(u);
}

/** Gamma(k, 1) sample for positive integer k, as a sum of k Exponentials. */
export function sampleGammaInt(rng, k) {
  let s = 0;
  for (let i = 0; i < k; i++) s += sampleExponential(rng);
  return s;
}

/** Beta(a, b) sample for positive integer a, b, via the Gamma ratio method. */
export function sampleBeta(rng, a, b) {
  const x = sampleGammaInt(rng, a);
  const y = sampleGammaInt(rng, b);
  return x / (x + y);
}
