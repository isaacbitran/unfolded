// A weight matrix as a glass slab with its real corner entries on the face
// (first 6 rows/cols, ⋯ / ⋮, last row/col), its name above and its true shape on the edges.

import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { Preview } from '../../model/trace';
import { GlyphField, layoutText, useAtlas, type GlyphFieldHandle, type GlyphItem } from '../glyphs/GlyphField';
import { Glass, type GlassHandle } from './Glass';
import { Label, fade } from './Label';

export interface MatrixSlabProps {
  preview: Preview;
  position: [number, number, number];
  /** face width/height in world units (shape is not to scale; the labels carry the truth) */
  w: number;
  h: number;
  d?: number;
  color: string;
  name: string; // KaTeX
  /** show the face numbers */
  numbers?: boolean;
  glyphH?: number;
}

export interface MatrixSlabHandle {
  setOpacity(a: number): void;
  setNumbersOpacity(a: number): void;
  setLabelOpacity(a: number): void;
  setBand(y: number, half: number, k: number): void;
}

const fmtShort = (v: number) => {
  const a = Math.abs(v);
  return (v < 0 ? '−' : '') + (a >= 10 ? a.toFixed(1) : a.toFixed(2));
};
const tnum = (n: number) => n.toLocaleString('en-US').replace(',', '{,}');

export const MatrixSlab = forwardRef<MatrixSlabHandle, MatrixSlabProps>(function MatrixSlab(
  { preview, position, w, h, d = 0.3, color, name, numbers = true, glyphH = 0.22 },
  ref,
) {
  const atlas = useAtlas();
  const glass = useRef<GlassHandle>(null);
  const field = useRef<GlyphFieldHandle>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const [x, y, z] = position;
  const items = useMemo(() => {
    if (!atlas || !numbers) return [] as GlyphItem[];
    const c = new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.55);
    const out: GlyphItem[] = [];
    const nr = preview.rows.length; // 7: six + last
    const nc = preview.cols.length;
    const slotsX = nc + 1; // + ⋯ column
    const slotsY = nr + 1; // + ⋮ row
    const sx = (w - 0.5) / slotsX;
    const sy = (h - 0.4) / slotsY;
    const fz = z + d / 2 + 0.02;
    const colX = (j: number) => x - w / 2 + 0.25 + sx * (j + 0.5);
    const rowY = (i: number) => y + h / 2 - 0.2 - sy * (i + 0.5);
    for (let i = 0; i < nr; i++) {
      const ry = rowY(i < nr - 1 ? i : i + 1);
      for (let j = 0; j < nc; j++) {
        const cx = colX(j < nc - 1 ? j : j + 1);
        out.push(...layoutText(fmtShort(preview.data[i * nc + j]), cx + sx * 0.42, ry, fz, { h: glyphH, color: c, k: 0.7, align: 'right' }));
      }
      out.push(...layoutText('⋯', colX(nc - 1), ry, fz, { h: glyphH, color: c, k: 0.5 }));
    }
    for (let j = 0; j < nc + 1; j++) {
      if (j === nc - 1) continue;
      out.push(...layoutText('⋮', colX(j) + sx * 0.15, rowY(nr - 1), fz, { h: glyphH, color: c, k: 0.5 }));
    }
    return out;
  }, [atlas, numbers, preview, color, x, y, z, w, h, d, glyphH]);
  useImperativeHandle(ref, () => ({
    setOpacity(a) {
      glass.current?.setOpacity(a);
      field.current?.setOpacity(a * 0.8);
      labels.current.forEach((l) => fade(l, a));
    },
    setNumbersOpacity: (a) => field.current?.setOpacity(a),
    setLabelOpacity: (a) => labels.current.forEach((l) => fade(l, a)),
    setBand: (by, half, k) => glass.current?.setBand(by, half, k),
  }));
  const [R, Ccount] = preview.shape;
  return (
    <group>
      <Glass ref={glass} size={[w, h, d]} position={position} color={color} fill={0.004} rim={0.2} edge={0.55} />
      <GlyphField ref={field} items={items} />
      <Label ref={(e) => void (labels.current[0] = e)} position={[x, y + h / 2 + 0.75, z]} tex={name} size={26} color={color} />
      <Label ref={(e) => void (labels.current[1] = e)} position={[x, y + h / 2 + 0.28, z]} tex={`${tnum(Ccount)}\\text{ cols}`} size={12} color="#9a9ea6" />
      <Label ref={(e) => void (labels.current[2] = e)} position={[x - w / 2 - 0.2, y, z]} tex={`${tnum(R)}\\text{ rows}`} size={12} color="#9a9ea6" align="right" />
    </group>
  );
});
