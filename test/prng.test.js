import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, choice, bernoulli, shuffle, sampleBeta, sampleGammaInt } from '../src/prng.js';

test('mulberry32 is deterministic for a given seed', () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  const seqA = Array.from({ length: 10 }, () => a());
  const seqB = Array.from({ length: 10 }, () => b());
  assert.deepEqual(seqA, seqB);
});

test('mulberry32 differs across seeds', () => {
  const a = mulberry32(1);
  const b = mulberry32(2);
  const seqA = Array.from({ length: 10 }, () => a());
  const seqB = Array.from({ length: 10 }, () => b());
  assert.notDeepEqual(seqA, seqB);
});

test('mulberry32 stays within [0, 1)', () => {
  const rng = mulberry32(7);
  for (let i = 0; i < 1000; i++) {
    const x = rng();
    assert.ok(x >= 0 && x < 1, `${x} out of range`);
  }
});

test('choice always returns an element of the array', () => {
  const rng = mulberry32(3);
  const arr = ['a', 'b', 'c'];
  for (let i = 0; i < 50; i++) {
    assert.ok(arr.includes(choice(rng, arr)));
  }
});

test('bernoulli(p=0) is always 0 and bernoulli(p=1) is always 1', () => {
  const rng = mulberry32(9);
  for (let i = 0; i < 20; i++) {
    assert.equal(bernoulli(rng, 0), 0);
    assert.equal(bernoulli(rng, 1), 1);
  }
});

test('shuffle is a permutation and does not mutate the input', () => {
  const rng = mulberry32(11);
  const arr = [1, 2, 3, 4, 5];
  const out = shuffle(rng, arr);
  assert.deepEqual(arr, [1, 2, 3, 4, 5]); // unmutated
  assert.deepEqual([...out].sort(), [1, 2, 3, 4, 5]); // same elements
});

test('sampleGammaInt(k) is positive and grows with k on average', () => {
  const rng = mulberry32(5);
  const small = Array.from({ length: 500 }, () => sampleGammaInt(rng, 1));
  const large = Array.from({ length: 500 }, () => sampleGammaInt(rng, 10));
  const meanOf = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  assert.ok(meanOf(large) > meanOf(small) * 3);
});

test('sampleBeta stays within (0, 1) and centers near a/(a+b)', () => {
  const rng = mulberry32(13);
  const samples = Array.from({ length: 2000 }, () => sampleBeta(rng, 8, 2));
  for (const s of samples) assert.ok(s > 0 && s < 1);
  const meanS = samples.reduce((a, b) => a + b, 0) / samples.length;
  assert.ok(Math.abs(meanS - 0.8) < 0.03, `mean ${meanS} too far from 0.8`);
});
