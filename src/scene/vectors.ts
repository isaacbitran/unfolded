// Layout helpers for vectors drawn as bracketed columns: first 6 entries, ⋮, last entry.

import * as THREE from 'three';
import { L, fmt } from './theme';
import { layoutText, type GlyphItem, type TextOpts } from './glyphs/GlyphField';

export interface Entry {
  /** index into the full vector, or −1 for the ellipsis row */
  idx: number;
  text: string;
  value: number;
}

/** The visible rows of a length-n vector stored at `data[offset … offset+n)`. */
export function visibleEntries(data: ArrayLike<number>, offset: number, n: number, head = L.visible): Entry[] {
  const rows: Entry[] = [];
  for (let i = 0; i < head; i++) rows.push({ idx: i, text: fmt(data[offset + i]), value: data[offset + i] });
  rows.push({ idx: -1, text: '⋮', value: 0 });
  rows.push({ idx: n - 1, text: fmt(data[offset + n - 1]), value: data[offset + n - 1] });
  return rows;
}

/** RMS of a vector slice; brightness of each entry is relative to it. */
export function rms(data: ArrayLike<number>, offset: number, n: number) {
  let s = 0;
  for (let i = 0; i < n; i++) s += data[offset + i] ** 2;
  return Math.sqrt(s / n) || 1;
}

/** Brightness for a value given its vector's scale: dim small entries, let large ones glow. */
export const glow = (v: number, scale: number) => 0.58 + 0.62 * Math.min(1, Math.abs(v) / (2.4 * scale));

export interface ColumnOpts extends Omit<TextOpts, 'from' | 'delay'> {
  /** per-row flight origin (same length as entries) */
  from?: ([number, number, number] | undefined)[];
  delay?: (row: number) => [number, number];
  scale?: number;
}

/** Glyphs for one column; row r sits at y0 − r·rowH. Numbers are right-aligned on a decimal grid. */
export function columnGlyphs(entries: Entry[], x: number, y0: number, z: number, o: ColumnOpts): GlyphItem[] {
  const out: GlyphItem[] = [];
  entries.forEach((e, r) => {
    const y = y0 - r * L.rowH;
    const k = e.idx < 0 ? 0.8 : glow(e.value, o.scale ?? 1) * (o.k ?? 1);
    // center the ellipsis, right-align numbers so decimal points line up
    const align = e.idx < 0 ? 'center' : 'right';
    const ax = e.idx < 0 ? x : x + 0.36;
    out.push(...layoutText(e.text, ax, y, z, { ...o, k, align, from: o.from?.[r], delay: o.delay?.(r) }));
  });
  return out;
}

export const columnHeight = (rows = L.visible + 2) => (rows - 1) * L.rowH;

/** Bracket box for a column whose first row is at y0. */
export function columnBox(x: number, y0: number, rows = L.visible + 2, halfW = 0.68) {
  return { x0: x - halfW, x1: x + halfW, y0: y0 - columnHeight(rows) - 0.24, y1: y0 + 0.24 };
}

export const white = new THREE.Color('#f4eee2');
