// Step 2 — the embedding matrix. Each token id travels to its row of W_E (a 50,257-row glass
// slab, rows at their true relative height), the row lights up, its 768 numbers (first 6, ⋯, last)
// lift off the slab and fly into a column vector under the token.

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { GlyphField, layoutText, textWidth, useAtlas, type GlyphFieldHandle, type GlyphItem } from '../glyphs/GlyphField';
import { Glass, Lines, bracketPoints, type GlassHandle } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TokenHeader, type TokenHeaderHandle } from '../primitives/TokenHeader';
import { C, L, colX, easeInOut, env, fmtInt, seg } from '../theme';
import { useStep } from '../useStep';
import { columnBox, columnGlyphs, rms, visibleEntries } from '../vectors';

export const SLAB = { x: -11.5, top: 5.2, h: 40, w: 7.4, d: 0.3, z: -1.6 };
export const COL_TOP = 1.25;
const FACE_Z = SLAB.z + SLAB.d / 2 + 0.02;
const VOCAB = 50257;

/** y of row `id` on the slab face (true fraction of the vocabulary, inside a small margin). */
export const rowY = (id: number) => SLAB.top - 0.7 - (SLAB.h - 1.4) * (id / (VOCAB - 1));
/** x of the j-th visible entry when a row is laid out horizontally on the face (8 slots). */
const rowSlotX = (j: number) => SLAB.x - SLAB.w / 2 + 0.75 + j * 0.84;

/** Timing for token k of T inside the step: start a, row flight window, etc. (all in step progress p). */
export function tokenWindow(k: number, T: number) {
  const span = 0.6; // p range over which starts are spread
  const a = 0.16 + (T > 1 ? (k * span) / (T - 1) : 0) * 0.86;
  const len = 0.2;
  return {
    idFly: [a, a + len * 0.35] as [number, number],
    band: [a + len * 0.3, a + len * 0.75] as [number, number],
    rowFly: [a + len * 0.55, a + len] as [number, number],
    end: a + len,
  };
}

export function S02Embedding({ index, trace }: { index: number; trace: SceneTrace }) {
  const atlas = useAtlas();
  const T = trace.T;
  const D = 768;
  const slab = useRef<GlassHandle>(null);
  const header = useRef<TokenHeaderHandle>(null);
  const flyRef = useRef<GlyphFieldHandle>(null);
  const idRef = useRef<GlyphFieldHandle>(null);
  const staticRef = useRef<GlyphFieldHandle>(null);
  const brackets = useRef<(THREE.LineSegments | null)[]>([]);
  const ticks = useRef<THREE.LineSegments>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);

  const windows = useMemo(() => trace.tokens.map((_, k) => tokenWindow(k, T)), [T, trace.tokens]);

  // Row numbers: start laid out horizontally on the slab face at the token's row, fly into the column.
  const flyItems = useMemo(() => {
    if (!atlas) return [];
    const items: GlyphItem[] = [];
    trace.tokens.forEach((tok, k) => {
      const entries = visibleEntries(trace.embedTok, k * D, D);
      const scale = rms(trace.embedTok, k * D, D);
      const w = windows[k];
      const y = rowY(tok.id);
      const from = entries.map((e, r): [number, number, number] | undefined => (e.idx < 0 ? undefined : [rowSlotX(r === entries.length - 1 ? 7 : r) + 0.25, y, FACE_Z + 0.02]));
      const col = columnGlyphs(entries, colX(k, T), COL_TOP, 0, {
        h: L.glyphH,
        color: new THREE.Color(C.token),
        scale,
        from,
        delay: (r) => {
          const t0 = w.rowFly[0] + r * 0.006;
          return [t0, w.rowFly[1] + r * 0.006];
        },
      });
      // ellipsis in the column appears as the numbers land; the face shows ⋯ instead
      col.forEach((g) => {
        if (g.ch === '⋮') {
          g.from = undefined;
          g.appear = w.rowFly[1] - 0.02;
        } else g.appear = w.band[0];
      });
      items.push(...col);
      items.push(...layoutText('⋯', rowSlotX(6) + 0.25, y, FACE_Z + 0.02, { h: L.glyphH, color: new THREE.Color(C.token), k: 0.7, appear: w.band[0], vanish: w.rowFly[0] + 0.03 }));
    });
    return items;
  }, [atlas, trace, T, windows]);

  // Token ids fly from under the label to the row index on the slab's left edge.
  const idItems = useMemo(() => {
    if (!atlas) return [];
    const items: GlyphItem[] = [];
    trace.tokens.forEach((tok, k) => {
      const w = windows[k];
      const s = String(tok.id);
      items.push(
        ...layoutText(s, SLAB.x - SLAB.w / 2 - 0.35, rowY(tok.id), FACE_Z, {
          h: 0.25,
          color: new THREE.Color(C.token),
          k: 1.25,
          align: 'right',
          from: [colX(k, T) + textWidth(s, 0.25) / 2, TOKEN_ID_Y, 0],
          delay: w.idFly,
        }),
      );
    });
    return items;
  }, [atlas, trace, T, windows]);

  // Static face texture: real rows of W_E sampled evenly through the vocabulary, at their true height.
  const staticItems = useMemo(() => {
    if (!atlas) return [];
    const p = trace.previews.W_E_grid;
    const items: GlyphItem[] = [];
    const color = new THREE.Color(C.token);
    p.rows.forEach((id, ri) => {
      const y = rowY(id);
      p.cols.forEach((_, ci) => {
        const slot = ci < 6 ? ci : 7;
        items.push(...layoutText(formatShort(p.data[ri * p.cols.length + ci]), rowSlotX(slot) + 0.25, y, FACE_Z, { h: 0.24, color, k: 0.62, align: 'right' }));
      });
      items.push(...layoutText('⋯', rowSlotX(6) + 0.25, y, FACE_Z, { h: 0.24, color, k: 0.45 }));
      items.push(...layoutText(String(id), SLAB.x - SLAB.w / 2 - 0.35, y, FACE_Z, { h: 0.2, color, k: 0.4, align: 'right' }));
    });
    return items;
  }, [atlas, trace]);

  const tickPoints = useMemo(() => {
    const pts: number[] = [];
    for (const tok of trace.tokens) {
      const y = rowY(tok.id);
      pts.push(SLAB.x - SLAB.w / 2, y, FACE_Z, SLAB.x + SLAB.w / 2, y, FACE_Z);
    }
    return pts;
  }, [trace]);

  const bracketPts = useMemo(
    () =>
      trace.tokens.map((_, k) => {
        const b = columnBox(colX(k, T), COL_TOP);
        return bracketPoints(b.x0, b.x1, b.y0, b.y1, 0);
      }),
    [trace, T],
  );

  useStep(index, (p, vis) => {
    const on = vis > 0.001;
    const slabA = vis * easeInOut(seg(p, 0, 0.12));
    slab.current?.setOpacity(slabA);
    staticRef.current?.setOpacity(slabA * (0.55 - 0.3 * seg(p, 0.12, 0.2)));
    header.current?.setOpacity(vis);
    header.current?.setIdOpacity(vis);
    flyRef.current?.setProgress(p);
    flyRef.current?.setOpacity(on ? vis : 0);
    flyRef.current?.setArc(1.2);
    idRef.current?.setProgress(p);
    idRef.current?.setOpacity(on ? vis : 0);
    idRef.current?.setArc(0.6);

    // band: the token currently being looked up
    let band = 0;
    let bandY = 0;
    windows.forEach((w, k) => {
      const e = env(p, w.band[0] - 0.02, w.rowFly[1], 0.04);
      if (e > band) {
        band = e;
        bandY = rowY(trace.tokens[k].id) - (SLAB.top - SLAB.h / 2);
      }
    });
    slab.current?.setBand(bandY, 0.22, band * 0.2);
    const lastBand = windows.length ? windows[0].band[0] : 0;
    if (ticks.current) {
      (ticks.current.material as THREE.LineBasicMaterial).opacity = 0.35 * vis * seg(p, lastBand, lastBand + 0.05);
      ticks.current.visible = on;
    }
    brackets.current.forEach((b, k) => {
      if (!b) return;
      (b.material as THREE.LineBasicMaterial).opacity = 0.75 * vis * seg(p, windows[k].rowFly[1] - 0.04, windows[k].rowFly[1]);
      b.visible = on;
    });
    fade(labels.current[0], vis * seg(p, 0.03, 0.12));
    fade(labels.current[1], vis * seg(p, 0.05, 0.14));
    fade(labels.current[2], vis * seg(p, 0.05, 0.14));
    fade(labels.current[3], vis * seg(p, 0.86, 0.95));
    trace.tokens.forEach((_, k) => fade(labels.current[4 + k], vis * seg(p, windows[k].rowFly[1], windows[k].rowFly[1] + 0.04)));
  });

  return (
    <group>
      <Glass ref={slab} size={[SLAB.w, SLAB.h, SLAB.d]} position={[SLAB.x, SLAB.top - SLAB.h / 2, SLAB.z]} color={C.token} fill={0.0025} rim={0.18} edge={0.42} />
      <Lines ref={ticks} points={tickPoints} color={C.token} opacity={0.35} />
      <GlyphField ref={staticRef} items={staticItems} />
      <GlyphField ref={idRef} items={idItems} />
      <GlyphField ref={flyRef} items={flyItems} />
      {bracketPts.map((pts, k) => (
        <Lines key={k} ref={(el) => void (brackets.current[k] = el)} points={pts} color={C.token} opacity={0} />
      ))}
      <TokenHeader ref={header} trace={trace} y={TOKEN_LABEL_Y} idY={TOKEN_ID_Y} />
      <Label ref={(el) => void (labels.current[0] = el)} position={[SLAB.x, SLAB.top + 1.1, SLAB.z]} tex="W_E" size={34} />
      <Label ref={(el) => void (labels.current[1] = el)} position={[SLAB.x, SLAB.top + 0.45, SLAB.z]} tex={`768 \\text{ columns}`} size={15} color="#a9adb5" />
      <Label ref={(el) => void (labels.current[2] = el)} position={[SLAB.x + SLAB.w / 2 + 0.5, SLAB.top - SLAB.h / 2, SLAB.z]} tex={`${fmtInt(VOCAB).replace(',', '{,}')} \\text{ rows}`} size={15} color="#a9adb5" align="left" />
      <Label ref={(el) => void (labels.current[3] = el)} position={[0, columnBox(0, COL_TOP).y0 - 0.75, 0]} tex={`X \\in \\mathbb{R}^{${T} \\times 768}`} size={20} />
      {trace.tokens.map((_, k) => (
        <Label key={k} ref={(el) => void (labels.current[4 + k] = el)} position={[colX(k, T), columnBox(0, COL_TOP).y0 - 0.22, 0]} tex="768" size={12} color="#8d929a" />
      ))}
    </group>
  );
}

export const TOKEN_LABEL_Y = 2.75;
export const TOKEN_ID_Y = 2.12;

function formatShort(v: number) {
  const a = Math.abs(v);
  return (v < 0 ? '−' : '') + (a >= 10 ? a.toFixed(1) : a.toFixed(2));
}
