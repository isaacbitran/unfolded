// Global camera path in story space s ∈ [0, N): each step's keys sit at s = stepIndex + key.p.
// Positions and look targets follow a C¹ cubic Hermite spline through the keys (finite-difference
// tangents over non-uniform knots), so the camera never stops dead or kinks at a step boundary.

import { STEPS, type CamKey, type V3 } from './steps';

interface Knot {
  s: number;
  pos: V3;
  look: V3;
  fov: number;
}

export function buildKnots(steps = STEPS): Knot[] {
  const knots: Knot[] = [];
  let fov = 34;
  steps.forEach((st, i) =>
    st.cam.forEach((k: CamKey) => {
      fov = k.fov ?? fov;
      knots.push({ s: i + k.p, pos: k.pos, look: k.look, fov });
    }),
  );
  return knots.sort((a, b) => a.s - b.s);
}

function hermite(p0: number, p1: number, m0: number, m1: number, t: number) {
  const t2 = t * t;
  const t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1;
}

function slope(knots: Knot[], i: number, get: (k: Knot) => number) {
  const a = knots[Math.max(0, i - 1)];
  const b = knots[Math.min(knots.length - 1, i + 1)];
  if (a === b) return 0;
  return (get(b) - get(a)) / (b.s - a.s);
}

export interface CamPose {
  pos: V3;
  look: V3;
  fov: number;
}

export function cameraAt(s: number, knots = defaultKnots): CamPose {
  if (s <= knots[0].s) return { pos: knots[0].pos, look: knots[0].look, fov: knots[0].fov };
  const last = knots[knots.length - 1];
  if (s >= last.s) return { pos: last.pos, look: last.look, fov: last.fov };
  let i = 0;
  while (knots[i + 1].s < s) i++;
  const k0 = knots[i];
  const k1 = knots[i + 1];
  const h = k1.s - k0.s;
  const t = h > 0 ? (s - k0.s) / h : 1;
  const comp = (get: (k: Knot) => number) => hermite(get(k0), get(k1), slope(knots, i, get) * h, slope(knots, i + 1, get) * h, t);
  const v = (which: 'pos' | 'look', c: number) => comp((k) => k[which][c]);
  return {
    pos: [v('pos', 0), v('pos', 1), v('pos', 2)],
    look: [v('look', 0), v('look', 1), v('look', 2)],
    fov: comp((k) => k.fov),
  };
}

const defaultKnots = buildKnots();
