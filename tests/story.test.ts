import { describe, expect, it } from 'vitest';
import { STEPS } from '../src/story/steps';
import { buildKnots, cameraAt } from '../src/story/camera';
import { stepVisibility } from '../src/scene/useStep';
import { env, seg } from '../src/scene/theme';
import { tokenWindow } from '../src/scene/steps/embedTiming';

const dist = (a: number[], b: number[]) => Math.hypot(...a.map((x, i) => x - b[i]));

describe('walkthrough story', () => {
  it('has the 17 storyboard steps, each with prose, shapes and camera keys', () => {
    expect(STEPS).toHaveLength(17);
    for (const s of STEPS) {
      expect(s.body.length).toBeGreaterThan(100);
      expect(s.question.endsWith('?')).toBe(true);
      expect(s.shapes.length).toBeGreaterThan(0);
      expect(s.cam.length).toBeGreaterThan(0);
      for (const k of s.cam) expect(k.p).toBeGreaterThanOrEqual(0), expect(k.p).toBeLessThanOrEqual(1);
    }
  });

  it('camera path is continuous: no jumps between consecutive frames anywhere in the story', () => {
    const N = STEPS.length;
    let prev = cameraAt(0);
    for (let s = 0.002; s < N; s += 0.002) {
      const c = cameraAt(s);
      // 0.002 of a step is ~1/300 of a scroll screen; even the long fly-throughs move < 2.5 units in it
      if (dist(c.pos, prev.pos) > 2.5 || dist(c.look, prev.look) > 2.5) throw new Error(`jump at s=${s.toFixed(3)}: ${dist(c.pos, prev.pos).toFixed(2)} / ${dist(c.look, prev.look).toFixed(2)}`);
      prev = c;
    }
  });

  it('camera settles exactly on every key', () => {
    for (const k of buildKnots()) {
      const c = cameraAt(k.s);
      expect(dist(c.pos, k.pos)).toBeLessThan(1e-9);
    }
  });

  it('crossfade: exactly one step fully visible mid-step, neighbours overlap at boundaries', () => {
    for (let k = 0; k < STEPS.length; k++) {
      expect(stepVisibility(k, k + 0.5)).toBe(1);
      expect(stepVisibility(k + 1, k + 0.5)).toBe(0);
    }
    const b = stepVisibility(3, 4.03) + stepVisibility(4, 4.03);
    expect(b).toBeGreaterThan(0.9);
  });

  it('choreography helpers are clamped and monotone', () => {
    expect(seg(-1, 0, 1)).toBe(0);
    expect(seg(2, 0, 1)).toBe(1);
    expect(seg(0.25, 0, 0.5)).toBe(0.5);
    expect(env(0.5, 0, 1)).toBe(1);
    expect(env(0, 0, 1)).toBe(0);
  });

  it('embedding lookups are sequential and finish inside the step', () => {
    for (const T of [1, 6, 20]) {
      const w = Array.from({ length: T }, (_, k) => tokenWindow(k, T));
      for (let k = 1; k < T; k++) expect(w[k].idFly[0]).toBeGreaterThan(w[k - 1].idFly[0]);
      for (const x of w) {
        expect(x.idFly[0]).toBeLessThan(x.band[0]);
        expect(x.band[0]).toBeLessThan(x.rowFly[1]);
        expect(x.end).toBeLessThanOrEqual(0.92);
      }
    }
  });
});
