// Step 16 — training. At the head end, every position (not just the last) shows its own next-token
// distribution (its top 3 predictions as bars). The token that actually follows each position flies
// down from the next column into that position's target slot, with the real probability the model
// gave it; a short flash of the causal mask shows that position i only ever saw tokens 1..i. The
// per-position losses −log p grow as bars, then flow into their mean, the training loss L. The last
// position has no target in this sentence, but its prediction is the one sampled at inference.

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { GlyphField, layoutText, useAtlas, type GlyphFieldHandle, type GlyphItem } from '../glyphs/GlyphField';
import { Glass, Lines, type GlassHandle } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TextPlane, measureText, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { C, easeInOut, lerp, seg } from '../theme';
import { useStep } from '../useStep';
import { Quads, rgba, type QuadItem, type QuadsHandle } from './S11MLP';
import { HZ } from './S13Unembed';

const SP = 2.3; // column spacing
const TOK_Y = 3.3;
const PRED_Y = [2.55, 2.2, 1.85];
const PRED_H = 0.19;
const PRED_K = 2.4; // bar length per unit probability
const SLOT_Y = 0.95;
const PROB_Y = 0.45;
const BASE_Y = -3.4;
const LOSS_K = 0.36; // bar height per nat
const CELL = 0.3;

const pctTex = (x: number) => (x >= 0.01 ? (100 * x).toFixed(1) : x >= 0.0001 ? (100 * x).toFixed(3) : (100 * x).toExponential(1)) + '\\%';

export function S16Training({ index, trace }: { index: number; trace: SceneTrace }) {
  const atlas = useAtlas();
  const T = trace.T;
  const xs = useMemo(() => trace.tokens.map((_, i) => (i - (T - 1) / 2) * SP), [trace, T]);
  const losses = useMemo(() => trace.positions.slice(0, T - 1).map((q) => -Math.log(Math.max(1e-12, q.pNext ?? 1))), [trace, T]);
  const meanLoss = useMemo(() => losses.reduce((a, b) => a + b, 0) / Math.max(1, losses.length), [losses]);
  const meanY = BASE_Y + meanLoss * LOSS_K;
  const L_X = xs[T - 2] + 0.75;
  const grid = { x0: xs[T - 1] - (T * CELL) / 2, y0: -0.55 };

  const toks = useRef<(TextPlaneHandle | null)[]>([]);
  const preds = useRef<(TextPlaneHandle | null)[]>([]);
  const predBars = useRef<QuadsHandle>(null);
  const targets = useRef<(TextPlaneHandle | null)[]>([]);
  const targetGroups = useRef<(THREE.Group | null)[]>([]);
  const lossBars = useRef<QuadsHandle>(null);
  const lossVals = useRef<GlyphFieldHandle>(null);
  const flow = useRef<GlyphFieldHandle>(null);
  const meanLine = useRef<THREE.LineSegments>(null);
  const maskCells = useRef<QuadsHandle>(null);
  const maskHi = useRef<QuadsHandle>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const focus = useRef<GlassHandle>(null);

  const predItems = useMemo(() => {
    const items: QuadItem[] = [];
    trace.positions.forEach((q, i) =>
      q.top.slice(0, 3).forEach((c, j) => {
        const w = Math.max(0.03, c.p * PRED_K);
        items.push({ x: xs[i] + 0.15 + w / 2, y: PRED_Y[j], z: HZ, w, h: 0.13, a: rgba(C.logit, j === 0 ? 0.95 : 0.55) });
      }),
    );
    return items;
  }, [trace, xs]);
  const lossItems = useMemo(
    () => losses.map((l, i): QuadItem => ({ x: xs[i], y: BASE_Y + (l * LOSS_K) / 2, z: HZ, w: 0.42, h: l * LOSS_K, from: [xs[i], BASE_Y, HZ], fromSize: [0.42, 0], a: rgba(C.token, 0.12 + 0.035 * l), delay: [0.56 + i * 0.02, 0.64 + i * 0.02] })),
    [losses, xs],
  );
  const lossGlyphs = useMemo(() => {
    if (!atlas) return { vals: [] as GlyphItem[], flow: [] as GlyphItem[] };
    const c = new THREE.Color(C.token);
    const vals: GlyphItem[] = [];
    const fl: GlyphItem[] = [];
    losses.forEach((l, i) => {
      const s = l.toFixed(2);
      const y = BASE_Y + l * LOSS_K + 0.25;
      vals.push(...layoutText(s, xs[i], y, HZ, { h: 0.24, color: c, k: 1.1, appear: 0.62 + i * 0.02 }));
      fl.push(...layoutText(s, L_X + 0.9, meanY + 0.02, HZ, { h: 0.24, color: c, k: 1.2, from: [xs[i], y, HZ], delay: [0.74 + i * 0.012, 0.82 + i * 0.012], appear: 0.7, vanish: 0.84 + i * 0.012 }));
    });
    return { vals, flow: fl };
  }, [atlas, losses, xs, L_X, meanY]);
  const maskItems = useMemo(() => {
    const items: QuadItem[] = [];
    for (let r = 0; r < T; r++)
      for (let k = 0; k < T; k++)
        items.push({ x: grid.x0 + (k + 0.5) * CELL, y: grid.y0 - (r + 0.5) * CELL, z: HZ, w: CELL * 0.86, h: CELL * 0.86, a: k <= r ? rgba(C.attn, 0.7) : rgba(C.dim, 0.08) });
    return items;
  }, [T, grid.x0, grid.y0]);
  const hiItems = useMemo((): QuadItem[] => [{ x: grid.x0 + (T * CELL) / 2, y: grid.y0 - CELL / 2, z: HZ - 0.01, w: T * CELL + 0.12, h: CELL + 0.04, a: rgba(C.token, 0.5, 0.22) }], [T, grid.x0, grid.y0]);
  const meanPts = useMemo(() => [xs[0] - 0.45, meanY, HZ, L_X, meanY, HZ], [xs, meanY, L_X]);

  const nextW = useMemo(() => trace.tokens.map((t) => measureText(visibleToken(t.text), 0.24)), [trace]);

  useStep(index, (sp, vis) => {
    const p = Math.min(1, sp / 0.88);
    const on = vis > 0.001;
    const infer = seg(p, 0.84, 0.92);
    // mask sweep: row r is "current" during its window
    const sweep = seg(p, 0.44, 0.56) * T;
    const maskOn = seg(p, 0.42, 0.45) * (1 - seg(p, 0.57, 0.6));
    const row = Math.min(T - 1, Math.floor(sweep));
    for (let i = 0; i < T; i++) {
      const hidden = maskOn > 0 && i > row ? 0.8 * maskOn : 0;
      const dimInfer = i < T - 1 ? 0.45 * infer : 0;
      toks.current[i]?.setOpacity(vis * seg(p, 0.04, 0.12) * (1 - hidden) * (1 - dimInfer));
    }
    // prediction panels
    predBars.current?.setOpacity(vis * seg(p, 0.12, 0.2));
    trace.positions.forEach((_, i) => {
      const a = seg(p, 0.1 + i * 0.025, 0.18 + i * 0.025);
      const dimInfer = i < T - 1 ? 0.55 * infer : 0;
      for (let j = 0; j < 3; j++) preds.current[i * 3 + j]?.setOpacity(vis * a * (j === 0 ? 1 : 0.7) * (1 - dimInfer));
    });
    // targets fly from the next column's token to this column's slot
    for (let i = 0; i < T - 1; i++) {
      const m = easeInOut(seg(p, 0.28 + i * 0.022, 0.36 + i * 0.022));
      const g = targetGroups.current[i];
      if (g) g.position.set(lerp(xs[i + 1], xs[i], m) - nextW[i + 1] / 2, lerp(TOK_Y, SLOT_Y, m), HZ + Math.sin(Math.PI * m) * 0.8);
      targets.current[i]?.setOpacity(vis * seg(p, 0.27 + i * 0.022, 0.29 + i * 0.022) * (1 - 0.45 * infer));
      fade(labels.current[i], vis * seg(p, 0.36 + i * 0.022, 0.4 + i * 0.022) * (1 - 0.45 * infer));
    }
    // loss bars, values, mean
    lossBars.current?.setProgress(p);
    lossBars.current?.setOpacity(vis * seg(p, 0.55, 0.56) * (1 - 0.5 * infer));
    lossVals.current?.setProgress(p);
    lossVals.current?.setOpacity(on ? vis * (1 - 0.5 * infer) : 0);
    flow.current?.setProgress(p);
    flow.current?.setArc(0.4);
    flow.current?.setOpacity(on ? vis : 0);
    if (meanLine.current) {
      const a = vis * seg(p, 0.72, 0.76);
      (meanLine.current.material as THREE.LineBasicMaterial).opacity = 0.6 * a;
      meanLine.current.visible = a > 0.002;
    }
    // mask
    maskCells.current?.setOpacity(vis * maskOn);
    if (on) maskHi.current?.setRect(0, grid.x0 + (T * CELL) / 2, grid.y0 - (row + 0.5) * CELL, T * CELL + 0.08, CELL);
    maskHi.current?.commit();
    maskHi.current?.setOpacity(vis * maskOn);

    focus.current?.setOpacity(vis * infer);
    const lab = (i: number, a: number) => fade(labels.current[T - 1 + i], vis * a);
    lab(0, seg(p, 0.36, 0.42) * (1 - seg(p, 0.42, 0.45)) + seg(p, 0.6, 0.64)); // no target
    lab(1, maskOn); // mask caption
    lab(2, seg(p, 0.82, 0.86)); // L
    lab(3, seg(p, 0.6, 0.66) * (1 - 0.5 * infer)); // loss caption
    lab(4, infer); // inference caption
    lab(5, seg(p, 0.14, 0.22) * (1 - 0.5 * infer)); // row caption: predictions
    lab(6, seg(p, 0.36, 0.42) * (1 - 0.5 * infer)); // row caption: target
  });

  return (
    <group>
      {trace.tokens.map((t, i) => (
        <TextPlane key={`t${i}`} ref={(e) => void (toks.current[i] = e)} text={visibleToken(t.text)} height={0.3} position={[xs[i], TOK_Y, HZ]} />
      ))}
      {trace.positions.flatMap((q, i) =>
        q.top.slice(0, 3).map((c, j) => (
          <TextPlane
            key={`p${i}-${j}`}
            ref={(e) => void (preds.current[i * 3 + j] = e)}
            text={visibleToken(c.text)}
            height={PRED_H}
            position={[xs[i] + 0.05 - measureText(visibleToken(c.text), PRED_H), PRED_Y[j], HZ]}
            anchor="left"
          />
        )),
      )}
      <Quads ref={predBars} items={predItems} />
      {trace.tokens.slice(1).map((t, i) => (
        <group key={`g${i}`} ref={(g) => void (targetGroups.current[i] = g)} position={[xs[i + 1], TOK_Y, HZ]}>
          <TextPlane ref={(e) => void (targets.current[i] = e)} text={visibleToken(t.text)} height={0.24} anchor="left" />
        </group>
      ))}
      <Quads ref={lossBars} items={lossItems} />
      <GlyphField ref={lossVals} items={lossGlyphs.vals} />
      <GlyphField ref={flow} items={lossGlyphs.flow} />
      <Lines ref={meanLine} points={meanPts} color={C.token} opacity={0} />
      <Quads ref={maskCells} items={maskItems} />
      <Quads ref={maskHi} items={hiItems} renderOrder={7} />
      <Glass ref={focus} size={[2.1, 3.9, 0.1]} position={[xs[T - 1], 1.75, HZ - 0.15]} color={C.token} fill={0.004} rim={0.1} edge={0.3} />

      {trace.positions.slice(0, T - 1).map((q, i) => (
        <Label key={`l${i}`} ref={(e) => void (labels.current[i] = e)} position={[xs[i], PROB_Y, HZ]} tex={`p=${pctTex(q.pNext ?? 0)}`} size={12} color="#a9adb5" />
      ))}
      <Label ref={(e) => void (labels.current[T - 1] = e)} position={[xs[T - 1], SLOT_Y, HZ]} tex={`\\text{no target}`} size={14} color="#8d929a" />
      <Label ref={(e) => void (labels.current[T] = e)} position={[xs[T - 1], grid.y0 + 0.3, HZ]} tex={`\\text{causal mask: row } i \\text{ sees } t_{\\le i}`} size={12} color="#cfd5dd" />
      <Label ref={(e) => void (labels.current[T + 1] = e)} position={[L_X + 0.15, meanY, HZ]} tex={`\\mathcal{L} = ${meanLoss.toFixed(2)}`} size={22} align="left" />
      <Label ref={(e) => void (labels.current[T + 2] = e)} position={[(xs[0] + xs[T - 2]) / 2, BASE_Y - 0.4, HZ]} tex={`-\\log p_\\theta(t_{i+1}\\mid t_{\\le i})\\ \\text{at each position}`} size={14} color="#a9adb5" />
      <Label ref={(e) => void (labels.current[T + 3] = e)} position={[xs[T - 1], PROB_Y, HZ]} tex={`\\text{sampled at inference}`} size={13} color="#f4eee2" />
      <Label ref={(e) => void (labels.current[T + 4] = e)} position={[xs[T - 1] + 1.25, PRED_Y[1], HZ]} tex={`\\text{top 3 predictions}`} size={12} color="#8d929a" align="left" />
      <Label ref={(e) => void (labels.current[T + 5] = e)} position={[xs[T - 1] + 1.25, SLOT_Y, HZ]} tex={`\\text{actual next token}`} size={12} color="#8d929a" align="left" />
    </group>
  );
}
