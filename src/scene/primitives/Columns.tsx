// A row of bracketed column vectors (one per token) built from a [T, D] tensor, with optional
// flight from another layout. The workhorse of most scenes.

import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { GlyphField, useAtlas, type GlyphFieldHandle, type GlyphItem } from '../glyphs/GlyphField';
import { L } from '../theme';
import { columnBox, columnGlyphs, rms, visibleEntries } from '../vectors';
import { Lines, bracketPoints } from './Glass';

export interface ColumnsProps {
  /** row-major [T, D] tensor */
  data: Float32Array;
  D: number;
  /** x of each column; length = number of columns to draw (rows of `data` used in order unless `rows` given) */
  xs: number[];
  rows?: number[];
  y0: number;
  z?: number;
  color: string;
  /** brightness multiplier */
  k?: number;
  /** column offset inside the row (e.g. head h → 64·h) and the length of the slice shown */
  offset?: number;
  /** length of the slice shown (defaults to D) */
  len?: number;
  /** flight origin per column: [x, y0, z] of where that column's first row starts */
  from?: ([number, number, number] | null)[];
  /** flight delay window per column */
  delay?: (col: number, row: number) => [number, number];
  /** visible leading entries */
  head?: number;
  glyphH?: number;
  brackets?: boolean;
}

export interface ColumnsHandle {
  setOpacity(a: number): void;
  setProgress(p: number): void;
  setArc(a: number): void;
  setBracketOpacity(a: number): void;
  field: GlyphFieldHandle | null;
}

export const Columns = forwardRef<ColumnsHandle, ColumnsProps>(function Columns(
  { data, D, xs, rows, y0, z = 0, color, k = 1, offset = 0, len: lenProp, from, delay, head = L.visible, glyphH = L.glyphH, brackets = true },
  ref,
) {
  const atlas = useAtlas();
  const field = useRef<GlyphFieldHandle>(null);
  const br = useRef<THREE.LineSegments>(null);
  const stride = D;
  const len = lenProp ?? D;
  const items = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const c = new THREE.Color(color);
    const out: GlyphItem[] = [];
    xs.forEach((x, ci) => {
      const r = rows ? rows[ci] : ci;
      const entries = visibleEntries(data, r * stride + offset, len, head);
      const scale = rms(data, r * stride + offset, len);
      const f = from?.[ci];
      out.push(
        ...columnGlyphs(entries, x, y0, z, {
          h: glyphH,
          color: c,
          k,
          scale,
          from: f ? entries.map((_, ri) => [f[0], f[1] - ri * L.rowH, f[2]] as [number, number, number]) : undefined,
          delay: delay ? (ri) => delay(ci, ri) : undefined,
        }),
      );
    });
    return out;
  }, [atlas, data, xs, rows, y0, z, color, k, offset, from, delay, head, glyphH, len, stride]);
  const pts = useMemo(() => xs.flatMap((x) => {
    const b = columnBox(x, y0, head + 2);
    return bracketPoints(b.x0, b.x1, b.y0, b.y1, z);
  }), [xs, y0, z, head]);
  useImperativeHandle(ref, () => ({
    field: field.current,
    setOpacity(a) {
      field.current?.setOpacity(a);
      if (br.current) {
        (br.current.material as THREE.LineBasicMaterial).opacity = 0.75 * a;
        br.current.visible = a > 0.002;
      }
    },
    setBracketOpacity(a) {
      if (br.current) {
        (br.current.material as THREE.LineBasicMaterial).opacity = 0.75 * a;
        br.current.visible = a > 0.002;
      }
    },
    setProgress: (p) => field.current?.setProgress(p),
    setArc: (a) => field.current?.setArc(a),
  }));
  return (
    <group>
      <GlyphField ref={field} items={items} />
      {brackets && <Lines ref={br} points={pts} color={color} opacity={0} />}
    </group>
  );
});
