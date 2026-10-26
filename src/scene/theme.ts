import * as THREE from 'three';

/** The color code from DESIGN.md §3. Nothing else in the scene is colored. */
export const C = {
  token: '#F4EEE2',
  position: '#F2B45C',
  norm: '#B9C3CF',
  query: '#5DA9FF',
  key: '#FF6B6B',
  value: '#5FD68A',
  attn: '#B48CFF',
  mlp: '#FFB454',
  logit: '#FFFFFF',
  dim: '#8a8f98',
} as const;
export type Role = keyof typeof C;

export const col = (hex: string) => new THREE.Color(hex);

/** Layout constants shared by every scene (world units). */
export const L = {
  colW: 2.0, // spacing between token columns
  rowH: 0.34, // vertical spacing of vector entries
  glyphH: 0.31, // glyph cell height
  visible: 6, // leading entries shown before the ellipsis
};

/** Token i of T, centered around x = 0. */
export const colX = (i: number, T: number) => (i - (T - 1) / 2) * L.colW;

export function fmt(v: number): string {
  if (!Number.isFinite(v)) return v > 0 ? '∞' : v < 0 ? '−∞' : 'NaN';
  const a = Math.abs(v);
  const s = a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a.toFixed(2);
  return (v < 0 && s !== '0.00' ? '−' : '') + s;
}

export const fmtInt = (n: number) => n.toLocaleString('en-US');

// ---------- easing / choreography helpers (pure; unit-tested) ----------
export const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
/** Local progress of p inside the window [a, b], clamped to 0..1. */
export const seg = (p: number, a: number, b: number) => clamp01((p - a) / (b - a));
export const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const easeOut = (t: number) => 1 - Math.pow(1 - t, 4);
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
/** 0 → 1 → 0 visibility envelope: fade in over [a, a+f], out over [b−f, b]. */
export const env = (p: number, a: number, b: number, f = 0.08) => Math.min(seg(p, a, a + f), 1 - seg(p, b - f, b));
