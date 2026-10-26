import { describe, expect, it } from 'vitest';
import { distribution, rng, sample } from '../src/model/sample';

const logits = new Float32Array([2, 1, 0.5, 0, -1, -3]);

describe('sampling', () => {
  it('softmax sums to 1 and is sorted', () => {
    const d = distribution(logits, { temperature: 1, topK: 0, topP: 1 }, 6);
    expect(d.reduce((s, c) => s + c.prob, 0)).toBeCloseTo(1, 6);
    expect(d.map((c) => c.id)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(d[5].cum).toBeCloseTo(1, 6);
  });
  it('temperature sharpens and flattens', () => {
    const cold = distribution(logits, { temperature: 0.5, topK: 0, topP: 1 }, 6);
    const hot = distribution(logits, { temperature: 2, topK: 0, topP: 1 }, 6);
    expect(cold[0].prob).toBeGreaterThan(hot[0].prob);
  });
  it('top-k keeps exactly k tokens and renormalises', () => {
    const d = distribution(logits, { temperature: 1, topK: 2, topP: 1 }, 6);
    expect(d.filter((c) => c.kept > 0).length).toBe(2);
    expect(d.reduce((s, c) => s + c.kept, 0)).toBeCloseTo(1, 6);
  });
  it('top-p keeps the smallest prefix whose mass reaches p', () => {
    const d = distribution(logits, { temperature: 1, topK: 0, topP: 0.8 }, 6);
    const kept = d.filter((c) => c.kept > 0);
    const massBeforeLast = kept.slice(0, -1).reduce((s, c) => s + c.prob, 0);
    expect(massBeforeLast).toBeLessThan(0.8);
    expect(kept.reduce((s, c) => s + c.prob, 0)).toBeGreaterThanOrEqual(0.8);
  });
  it('temperature 0 is greedy', () => {
    expect(sample(logits, { temperature: 0, topK: 0, topP: 1 }, rng(1)).id).toBe(0);
  });
  it('empirical frequencies follow the distribution', () => {
    const r = rng(42);
    const counts = new Array(6).fill(0);
    const p = { temperature: 1, topK: 0, topP: 1 };
    for (let i = 0; i < 20000; i++) counts[sample(logits, p, r).id]++;
    const d = distribution(logits, p, 6);
    for (const c of d) expect(Math.abs(counts[c.id] / 20000 - c.prob)).toBeLessThan(0.015);
  });
});
