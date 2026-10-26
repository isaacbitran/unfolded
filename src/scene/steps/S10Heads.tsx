// Step 10 — twelve heads, then W_O. First the real attention patterns of all 12 heads of layer 0
// as small heat grids in a shallow 4×3 arc, each captioned by a rule computed from its pattern
// (`headCaption`). Then the last token's 12 head outputs z^h (64 numbers each) fly out of their
// grids and stack into one 768-number column, which passes into the W_O slab; the attention
// output for every token flies out of the slab, rises onto the residual stream x⁰, and the sum
// x⁰ + attn(x) (layers[0].residMid) rolls into place.

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { GlyphField, layoutText, useAtlas, type GlyphFieldHandle, type GlyphItem } from '../glyphs/GlyphField';
import { Columns, type ColumnsHandle } from '../primitives/Columns';
import { Lines, bracketPoints } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { MatrixSlab, type MatrixSlabHandle } from '../primitives/MatrixSlab';
import { TokenHeader, type TokenHeaderHandle } from '../primitives/TokenHeader';
import { C, L, colX, easeInOut, fmt, seg } from '../theme';
import { useStep } from '../useStep';
import { columnBox } from '../vectors';
import { COL_TOP, TOKEN_ID_Y, TOKEN_LABEL_Y } from './S02Embedding';
import { HeatCells, focusHead, lin, type HeatCellsHandle } from './S08Softmax';

const PURPLE = lin(C.attn);

const D = 768;
const DH = 64;
const H = 12;

/**
 * A one-phrase description of a head's attention pattern, computed from the pattern alone.
 * Over query rows i = 2..T−1 (rows 0 and 1 are skipped because there every rule trivially
 * coincides; with T < 3 all rows i ≥ 1 are used) take the mean weight on
 *   - the previous token  A[i, i−1]  → "previous token"
 *   - the first token     A[i, 0]    → "first token"
 *   - the token itself    A[i, i]    → "self"
 * The largest of the three wins if it is at least 0.5 (the head puts most of its attention
 * there on average); otherwise the head spreads its attention and is called "broad".
 * The mean that decided the label is returned with it.
 */
export function headCaption(pattern: Float32Array, h: number, T: number): { label: string; share: number | null } {
  if (T < 2) return { label: 'self', share: 1 };
  const i0 = T >= 3 ? 2 : 1;
  const o = h * T * T;
  let prev = 0;
  let first = 0;
  let self = 0;
  for (let i = i0; i < T; i++) {
    prev += pattern[o + i * T + i - 1];
    first += pattern[o + i * T];
    self += pattern[o + i * T + i];
  }
  const n = T - i0;
  const cands: [string, number][] = [
    ['previous token', prev / n],
    ['first token', first / n],
    ['self', self / n],
  ];
  const [label, share] = cands.reduce((b, c) => (c[1] > b[1] ? c : b));
  return share >= 0.5 ? { label, share } : { label: 'broad', share: null };
}

const SLAB = { x: 12.0, y: -0.9, w: 3.7, h: 5.6 };
const STACK_X = 7.75;
const POS_TOP = COL_TOP - 3.55;

/** Timing (step progress). */
const TM = {
  gridsIn: [0.0, 0.1],
  captions: [0.05, 0.2],
  gridsOut: [0.36, 0.46],
  stackFly: [0.36, 0.5],
  colsIn: [0.42, 0.52],
  intoSlab: [0.58, 0.67],
  outOfSlab: [0.64, 0.79],
  merge: [0.78, 0.86],
  sum: [0.865, 0.95],
} as const;

export function S10Heads({ index, trace }: { index: number; trace: SceneTrace }) {
  const atlas = useAtlas();
  const T = trace.T;
  const last = T - 1;
  const L0 = trace.layers[0];
  const fh = useMemo(() => focusHead(trace), [trace]);
  const captions = useMemo(() => Array.from({ length: H }, (_, h) => headCaption(L0.pattern, h, T)), [L0, T]);
  const xs = useMemo(() => trace.tokens.map((_, k) => colX(k, T)), [trace, T]);

  // head grids on a shallow arc
  const g = Math.min(0.34, 2.05 / T);
  const G = g * T;
  const gridC = useMemo(
    () =>
      Array.from({ length: H }, (_, h) => {
        const c = (h % 4) - 1.5;
        const r = Math.floor(h / 4) - 1;
        return [c * 3.3, 0.9 - r * 3.2, -0.5 * c * c] as [number, number, number];
      }),
    [],
  );
  const centers = useMemo(() => {
    const out: [number, number, number][] = [];
    gridC.forEach(([gx, gy, gz]) => {
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) out.push([gx - G / 2 + g * (j + 0.5), gy + G / 2 - g * (i + 0.5), gz]);
    });
    return out;
  }, [gridC, T, g, G]);
  const frames = useMemo(
    () =>
      gridC.flatMap(([gx, gy, gz]) => {
        const a = gx - G / 2 - 0.04;
        const b = gx + G / 2 + 0.04;
        const t = gy + G / 2 + 0.04;
        const d = gy - G / 2 - 0.04;
        return [a, t, gz, b, t, gz, b, t, gz, b, d, gz, b, d, gz, a, d, gz, a, d, gz, a, t, gz];
      }),
    [gridC, G],
  );
  const focusFrame = useMemo(() => {
    const [gx, gy, gz] = gridC[fh];
    const a = gx - G / 2 - 0.14;
    const b = gx + G / 2 + 0.14;
    const t = gy + G / 2 + 0.14;
    const d = gy - G / 2 - 0.14;
    return [a, t, gz, b, t, gz, b, t, gz, b, d, gz, b, d, gz, a, d, gz, a, d, gz, a, t, gz];
  }, [gridC, G, fh]);

  // the stacked z_T column: one row per head, "first ⋯ last" of its 64 numbers
  const stackTop = COL_TOP + 0.3;
  const rowY = (h: number) => stackTop - h * L.rowH;
  const stackBox = useMemo(() => bracketPoints(STACK_X - 1.15, STACK_X + 1.15, rowY(H - 1) - 0.24, stackTop + 0.24, 0), []); // eslint-disable-line react-hooks/exhaustive-deps

  const flights = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const out: GlyphItem[] = [];
    const green = new THREE.Color(C.value);
    for (let h = 0; h < H; h++) {
      const o = last * D + h * DH;
      const txt = `${fmt(L0.z[o])} ⋯ ${fmt(L0.z[o + DH - 1])}`;
      const y = rowY(h);
      const [gx, gy, gz] = gridC[h];
      const a = TM.stackFly[0] + h * 0.006;
      const k = h === fh ? 1.15 : 0.85;
      // grid → stack
      out.push(
        ...layoutText(txt, STACK_X + 0.15, y, 0, {
          h: 0.27, color: green, k, align: 'center',
          from: [gx, gy, gz], delay: [a, a + 0.08], appear: a, vanish: TM.intoSlab[0] + 0.012 + h * 0.003,
        }),
      );
      // stack → into the slab face, converging toward its middle
      const b = TM.intoSlab[0] + h * 0.003;
      out.push(
        ...layoutText(txt, SLAB.x - SLAB.w / 2 + 0.9, SLAB.y + 1.6 - h * 0.28, 0.2, {
          h: 0.27, color: green, k, align: 'center',
          from: [STACK_X + 0.15, y, 0], delay: [b, b + 0.06], appear: b, vanish: b + 0.065,
        }),
      );
    }
    return out;
  }, [atlas, L0, last, gridC, fh]); // eslint-disable-line react-hooks/exhaustive-deps

  const statics = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const out: GlyphItem[] = [];
    const dim = new THREE.Color(C.dim);
    for (let h = 0; h < H; h++)
      out.push(...layoutText(String(h + 1), STACK_X - 1.45, rowY(h), 0, { h: 0.24, color: dim, k: 0.9, align: 'right', appear: TM.stackFly[1] - 0.04, vanish: TM.intoSlab[0] + 0.03 }));
    return out;
  }, [atlas]); // eslint-disable-line react-hooks/exhaustive-deps

  // attention output columns start at the slab face, one per token
  const outFrom = useMemo(() => xs.map((_, i) => [SLAB.x - 0.4, SLAB.y + SLAB.h / 2 - 0.9 - i * 0.5, 0.25] as [number, number, number]), [xs]);
  const outDelay = useMemo(() => (c: number, r: number) => [c * 0.06 + r * 0.004, 0.6 + c * 0.06 + r * 0.004] as [number, number], []);
  const sumFrom = useMemo(() => xs.map((x) => [x, COL_TOP - 0.22, 0] as [number, number, number]), [xs]);
  const sumDelay = useMemo(() => (c: number, r: number) => [c * 0.04 + r * 0.02, 0.4 + c * 0.04 + r * 0.02] as [number, number], []);

  const cells = useRef<HeatCellsHandle>(null);
  const frameRef = useRef<THREE.LineSegments>(null);
  const focusRef = useRef<THREE.LineSegments>(null);
  const stackBr = useRef<THREE.LineSegments>(null);
  const fly = useRef<GlyphFieldHandle>(null);
  const stat = useRef<GlyphFieldHandle>(null);
  const header = useRef<TokenHeaderHandle>(null);
  const xCols = useRef<ColumnsHandle>(null);
  const outCols = useRef<ColumnsHandle>(null);
  const sumCols = useRef<ColumnsHandle>(null);
  const slab = useRef<MatrixSlabHandle>(null);
  const outGroup = useRef<THREE.Group>(null);
  const capL = useRef<(HTMLDivElement | null)[]>([]);
  const labels = useRef<(HTMLDivElement | null)[]>([]);

  const lineOp = (l: THREE.LineSegments | null, a: number) => {
    if (!l) return;
    (l.material as THREE.LineBasicMaterial).opacity = a;
    l.visible = a > 0.002;
  };

  useStep(index, (p, v) => {
    // sharpened crossfade: this step's objects arrive late and leave early, so neighbours barely overlap
    const e = seg(v, 0.6, 1);
    const vis = e * e * (3 - 2 * e);
    const on = vis > 0.001;
    // phase A: head grids
    const gOut = 1 - easeInOut(seg(p, TM.gridsOut[0], TM.gridsOut[1]));
    const cl = cells.current;
    if (cl) {
      let n = 0;
      for (let h = 0; h < H; h++) {
        const a = vis * easeInOut(seg(p, TM.gridsIn[0] + h * 0.005, TM.gridsIn[1] - 0.05 + h * 0.005)) * gOut;
        const base = h * T * T;
        for (let q = 0; q < T * T; q++) {
          const w = L0.pattern[base + q];
          const upper = q % T > Math.floor(q / T);
          const k = upper ? 0.004 : 0.012 + 0.42 * Math.pow(w, 0.8);
          cl.set(n++, PURPLE[0] * k * a, PURPLE[1] * k * a, PURPLE[2] * k * a);
        }
      }
      cl.commit(on && gOut > 0.001);
    }
    lineOp(frameRef.current, 0.22 * vis * seg(p, 0.0, 0.08) * gOut);
    lineOp(focusRef.current, 0.9 * vis * seg(p, 0.22, 0.28) * gOut);
    capL.current.forEach((el, h) => fade(el, vis * gOut * seg(p, TM.captions[0] + h * 0.006, TM.captions[0] + 0.04 + h * 0.006)));

    // phase B: the z stack, residual columns and W_O
    fly.current?.setProgress(p);
    fly.current?.setOpacity(on ? vis : 0);
    fly.current?.setArc(0.9);
    stat.current?.setProgress(p);
    stat.current?.setOpacity(on ? vis : 0);
    lineOp(stackBr.current, 0.8 * vis * seg(p, TM.stackFly[1] - 0.04, TM.stackFly[1]) * (1 - seg(p, TM.intoSlab[0], TM.intoSlab[0] + 0.04)));
    const colsA = vis * easeInOut(seg(p, TM.colsIn[0], TM.colsIn[1]));
    header.current?.setOpacity(colsA);
    header.current?.setIdOpacity(0);
    header.current?.setFocus(last, 0.6 * seg(p, TM.colsIn[1], TM.colsIn[1] + 0.05) * (1 - seg(p, TM.outOfSlab[0], TM.outOfSlab[0] + 0.05)));
    const sumA = easeInOut(seg(p, TM.sum[0], TM.sum[0] + 0.05));
    const partsA = 1 - seg(p, TM.sum[0], TM.sum[0] + 0.03);
    xCols.current?.setOpacity(colsA * partsA * (1 - 0.6 * seg(p, TM.merge[0], TM.merge[1])));
    slab.current?.setOpacity(colsA * (1 - 0.6 * seg(p, TM.merge[0], TM.merge[1])));
    const band = Math.min(seg(p, TM.intoSlab[0], TM.intoSlab[1]), 1 - seg(p, TM.outOfSlab[0] + 0.04, TM.outOfSlab[1]));
    slab.current?.setBand(0, SLAB.h * 0.45, band * 0.35);
    outCols.current?.setProgress(seg(p, TM.outOfSlab[0], TM.outOfSlab[1]));
    outCols.current?.setArc(1.4);
    outCols.current?.setOpacity(vis * seg(p, TM.outOfSlab[0], TM.outOfSlab[0] + 0.03) * partsA);
    outCols.current?.setBracketOpacity(vis * seg(p, TM.outOfSlab[1] - 0.03, TM.outOfSlab[1]) * (1 - seg(p, TM.merge[0], TM.merge[0] + 0.03)));
    xCols.current?.setBracketOpacity(colsA * (1 - seg(p, TM.merge[0], TM.merge[0] + 0.03)));
    const merge = easeInOut(seg(p, TM.merge[0], TM.merge[1]));
    if (outGroup.current) outGroup.current.position.y = (COL_TOP - POS_TOP) * merge;
    sumCols.current?.setOpacity(vis * sumA);
    sumCols.current?.setProgress(seg(p, TM.sum[0], TM.sum[1]));

    fade(labels.current[0], vis * seg(p, TM.stackFly[1] - 0.03, TM.stackFly[1] + 0.01) * (1 - seg(p, TM.intoSlab[0], TM.intoSlab[0] + 0.03))); // stack name
    fade(labels.current[1], vis * seg(p, TM.stackFly[1] - 0.03, TM.stackFly[1] + 0.01) * (1 - seg(p, TM.intoSlab[0], TM.intoSlab[0] + 0.03))); // 768
    fade(labels.current[2], colsA * (1 - seg(p, TM.merge[0], TM.merge[0] + 0.04))); // x⁰
    fade(labels.current[3], vis * seg(p, TM.outOfSlab[1] - 0.04, TM.outOfSlab[1]) * (1 - seg(p, TM.merge[0], TM.merge[0] + 0.04))); // attn(x)
    xs.forEach((_, k) => fade(labels.current[4 + k], vis * seg(p, TM.outOfSlab[1] - 0.04, TM.outOfSlab[1]) * (1 - seg(p, TM.merge[0], TM.merge[0] + 0.02))));
    fade(labels.current[4 + T], vis * seg(p, TM.sum[1] - 0.04, TM.sum[1] + 0.01)); // final shape
    fade(labels.current[5 + T], vis * seg(p, TM.captions[0], TM.captions[0] + 0.06) * (1 - seg(p, 0.3, 0.34))); // phase-A title
  });

  const plusY = COL_TOP - (L.visible + 1) * L.rowH - 0.62;
  const bottom = columnBox(0, COL_TOP).y0;
  return (
    <group>
      <HeatCells ref={cells} centers={centers} size={g} gap={0.08} />
      <Lines ref={frameRef} points={frames} color={C.token} opacity={0} />
      <Lines ref={focusRef} points={focusFrame} color={C.attn} opacity={0} />
      {gridC.map(([gx, gy, gz], h) => (
        <Label
          key={h}
          ref={(e) => void (capL.current[h] = e)}
          position={[gx, gy - G / 2 - 0.42, gz]}
          text={`${h + 1} · ${captions[h].label}${captions[h].share !== null ? ` ${Math.round(captions[h].share! * 100)}%` : ''}`}
          size={13}
          color={h === fh ? C.attn : '#a9adb5'}
        />
      ))}

      <TokenHeader ref={header} trace={trace} y={TOKEN_LABEL_Y} idY={TOKEN_ID_Y} />
      <Columns ref={xCols} data={trace.resid[0]} D={D} xs={xs} y0={COL_TOP} color={C.token} />
      <group ref={outGroup}>
        <Columns ref={outCols} data={L0.attnOut} D={D} xs={xs} y0={POS_TOP} color={C.attn} from={outFrom} delay={outDelay} />
      </group>
      <Columns ref={sumCols} data={L0.residMid} D={D} xs={xs} y0={COL_TOP} color={C.token} from={sumFrom} delay={sumDelay} />
      <MatrixSlab ref={slab} preview={trace.previews.W_O} position={[SLAB.x, SLAB.y, 0]} w={SLAB.w} h={SLAB.h} color={C.attn} name="W_O" />
      <GlyphField ref={stat} items={statics} />
      <GlyphField ref={fly} items={flights} />
      <Lines ref={stackBr} points={stackBox} color={C.value} opacity={0} />

      <Label ref={(e) => void (labels.current[0] = e)} position={[STACK_X, stackTop + 0.75, 0]} tex={`[z^{1};\\dots;z^{12}]_T`} size={18} color={C.value} />
      <Label ref={(e) => void (labels.current[1] = e)} position={[STACK_X, rowY(H - 1) - 0.6, 0]} tex={`12\\times 64 = 768`} size={13} color="#9a9ea6" />
      <Label ref={(e) => void (labels.current[2] = e)} position={[xs[0] - 1.0, COL_TOP - 1.2, 0]} tex={`x^{0}`} size={18} color={C.token} align="right" />
      <Label ref={(e) => void (labels.current[3] = e)} position={[xs[0] - 1.0, POS_TOP - 1.2, 0]} tex={`\\text{attn}(x)`} size={18} color={C.attn} align="right" />
      {xs.map((x, k) => (
        <Label key={`p${k}`} ref={(e) => void (labels.current[4 + k] = e)} position={[x, plusY, 0]} tex="+" size={22} color="#d8d4ca" />
      ))}
      <Label ref={(e) => void (labels.current[4 + T] = e)} position={[0, bottom - 0.75, 0]} tex={`x^{0} + \\text{attn}(x) \\in \\mathbb{R}^{${T}\\times 768}`} size={20} />
      <Label ref={(e) => void (labels.current[5 + T] = e)} position={[0, gridC[0][1] + G / 2 + 0.85, 0]} tex={`A_1,\\ \\dots,\\ A_{12} \\qquad \\text{block 1}`} size={18} />
    </group>
  );
}
