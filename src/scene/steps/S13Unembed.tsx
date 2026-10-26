// Step 13 — from vector to vocabulary, at the head end of the tower. Only the last token's column
// continues: x¹²_T passes the final LayerNorm (its real μ and σ), flies into the unembedding slab
// (the same table as W_E, tied, so labelled W_Eᵀ), and a scan line sweeps down all 50,257 rows.
// As it passes, each row's dot product appears as a bar in a tall strip of logits beside the slab
// (rows at their true vocabulary position; one bar per pixel-height bin shows the bin's maximum).
// The ten largest logits get their own bright bars and labels with their token text.

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { topIndices } from '../../model/sample';
import type { SceneTrace } from '../../model/trace';
import { GlyphField, layoutText, useAtlas, type GlyphFieldHandle, type GlyphItem } from '../glyphs/GlyphField';
import { Columns, type ColumnsHandle } from '../primitives/Columns';
import { Glass, Lines, type GlassHandle } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TextPlane, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { C, easeInOut, fmt, seg } from '../theme';
import { useStep } from '../useStep';
import { columnBox } from '../vectors';
import { Quads, rgba, type QuadItem, type QuadsHandle } from './S11MLP';

const VOCAB = 50257;
/** z of the head-end plane (steps 13, 14 and 16) */
export const HZ = -206; // just past the end of the tower's rails (z = −178), so the camera is never inside block 12
const COL_TOP = 1.7;
const X_COL = -9.2;
const LN_COL = -6.3;
export const WU = { x: -2.4, top: 3.6, h: 8.8, w: 3.0, d: 0.3 };
const FACE_Z = HZ + WU.d / 2 + 0.02;
/** the logit strip: bars start at x0 and are at most len long */
export const STRIP = { x0: -0.55, len: 4.4, bins: 1024 };
const LABEL_X = 4.7;

/** y of vocabulary row `id` on the slab and in the strip */
export const vocabY = (id: number) => WU.top - 0.25 - (WU.h - 0.5) * (id / (VOCAB - 1));

/** min and max of the last position's logits (the strip's scale) */
export function logitRange(l: Float32Array) {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < l.length; i++) {
    if (l[i] < lo) lo = l[i];
    if (l[i] > hi) hi = l[i];
  }
  return { lo, hi };
}
/** bar length of a logit in the strip */
export const stripLen = (v: number, lo: number, hi: number) => STRIP.len * ((v - lo) / (hi - lo));

const fmtShort = (v: number) => {
  const a = Math.abs(v);
  return (v < 0 ? '−' : '') + (a >= 10 ? a.toFixed(1) : a.toFixed(2));
};

/** Spread label ys (sorted top to bottom) so neighbours are at least `gap` apart. */
export function spreadLabels(ys: number[], gap: number): number[] {
  const out = ys.slice();
  for (let i = 1; i < out.length; i++) if (out[i - 1] - out[i] < gap) out[i] = out[i - 1] - gap;
  // centre the block on the original mean so it does not drift downward
  const shift = (ys.reduce((a, b) => a + b, 0) - out.reduce((a, b) => a + b, 0)) / out.length;
  return out.map((y) => y + shift);
}

/** Where the ten largest logits sit in the strip and where their labels go (shared with step 14). */
export function topLabelLayout(logits: Float32Array, top: number[], lo: number, hi: number) {
  const sorted = top.slice().sort((a, b) => a - b); // by vocabulary row, top to bottom
  const ys = spreadLabels(sorted.map(vocabY), 0.34);
  return sorted.map((id, i) => ({ id, y: vocabY(id), ly: ys[i], len: stripLen(logits[id], lo, hi), rank: top.indexOf(id) }));
}
/** x where step 13's top-10 token names start */
export const NAME_X = 4.7 + 1.15;
export const NAME_H = 0.22;

export function S13Unembed({ index, trace }: { index: number; trace: SceneTrace }) {
  const atlas = useAtlas();
  const T = trace.T;
  const t = T - 1;
  const logits = trace.logitsLast;
  const { lo, hi } = useMemo(() => logitRange(logits), [logits]);
  const top = useMemo(() => topIndices(logits, 10), [logits]);

  const xCol = useRef<ColumnsHandle>(null);
  const lnCol = useRef<ColumnsHandle>(null);
  const lnFly = useRef<ColumnsHandle>(null);
  const slab = useRef<GlassHandle>(null);
  const face = useRef<GlyphFieldHandle>(null);
  const strip = useRef<QuadsHandle>(null);
  const topBars = useRef<QuadsHandle>(null);
  const values = useRef<GlyphFieldHandle>(null);
  const leaders = useRef<THREE.LineSegments>(null);
  const scan = useRef<THREE.LineSegments>(null);
  const tok = useRef<TextPlaneHandle>(null);
  const names = useRef<(TextPlaneHandle | null)[]>([]);
  const labels = useRef<(HTMLDivElement | null)[]>([]);

  const SWEEP: [number, number] = [0.42, 0.72];

  // strip bins: maximum logit per bin, revealed as the scan line passes
  const stripItems = useMemo(() => {
    const items: QuadItem[] = [];
    const per = Math.ceil(VOCAB / STRIP.bins);
    const binH = (WU.h - 0.5) / STRIP.bins;
    for (let b = 0; b < STRIP.bins; b++) {
      let m = -Infinity;
      for (let i = b * per; i < Math.min(VOCAB, (b + 1) * per); i++) m = Math.max(m, logits[i]);
      if (!Number.isFinite(m)) continue;
      const id = Math.min(VOCAB - 1, b * per + per / 2);
      const f = (b + 0.5) / STRIP.bins;
      const n = (m - lo) / (hi - lo);
      const len = Math.max(0.01, stripLen(m, lo, hi));
      const y = vocabY(id);
      const d0 = SWEEP[0] + (SWEEP[1] - SWEEP[0]) * f;
      items.push({ x: STRIP.x0 + len / 2, y, z: HZ, w: len, h: binH * 1.05, from: [STRIP.x0, y, HZ], fromSize: [0, binH], a: rgba(C.logit, 0.03 + 0.26 * n * n * n * n), delay: [d0, d0 + 0.035] });
    }
    return items;
  }, [logits, lo, hi]);

  // the ten largest logits as individual bright bars
  const topLayout = useMemo(() => topLabelLayout(logits, top, lo, hi), [top, logits, lo, hi]);
  const topItems = useMemo(
    () =>
      topLayout.map((l): QuadItem => {
        const d0 = SWEEP[0] + (SWEEP[1] - SWEEP[0]) * (l.id / VOCAB);
        return { x: STRIP.x0 + l.len / 2, y: l.y, z: HZ + 0.01, w: l.len, h: 0.035, from: [STRIP.x0, l.y, HZ + 0.01], fromSize: [0, 0.035], a: rgba(C.logit, 0.7), b: rgba(C.logit, 1.6), delay: [d0, d0 + 0.035], key: 0.05 + l.rank * 0.06 };
      }),
    [topLayout],
  );
  const leaderPts = useMemo(
    () => topLayout.flatMap((l) => [STRIP.x0 + l.len + 0.08, l.y, HZ, LABEL_X - 0.12, l.ly, HZ]),
    [topLayout],
  );
  const valueItems = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const c = new THREE.Color(C.dim);
    return topLayout.flatMap((l) => layoutText(fmt(logits[l.id]), LABEL_X + 0.95, l.ly, HZ, { h: 0.21, color: c, k: 1.4, align: 'right' }));
  }, [atlas, topLayout, logits]);

  // faint real rows of W_E on the slab face, at their true vocabulary height
  const faceItems = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const p = trace.previews.W_E_grid;
    const items: GlyphItem[] = [];
    const color = new THREE.Color(C.token);
    const slotX = (j: number) => WU.x - WU.w / 2 + 0.55 + j * 0.5;
    p.rows.forEach((id, ri) => {
      const y = vocabY(id);
      [0, 1, 2].forEach((ci, j) => items.push(...layoutText(fmtShort(p.data[ri * p.cols.length + ci]), slotX(j) + 0.2, y, FACE_Z, { h: 0.15, color, k: 0.6, align: 'right' })));
      items.push(...layoutText('⋯', slotX(3) + 0.05, y, FACE_Z, { h: 0.15, color, k: 0.45 }));
      items.push(...layoutText(fmtShort(p.data[ri * p.cols.length + p.cols.length - 1]), slotX(4) + 0.2, y, FACE_Z, { h: 0.15, color, k: 0.6, align: 'right' }));
    });
    return items;
  }, [atlas, trace]);

  const xs = useMemo(() => [X_COL], []);
  const lnXs = useMemo(() => [LN_COL], []);
  const rows = useMemo(() => [t], [t]);
  const lnFrom = useMemo(() => [[X_COL, COL_TOP, HZ] as [number, number, number]], []);
  const lnDelay = useMemo(() => (_c: number, r: number) => [0.18 + r * 0.008, 0.3 + r * 0.008] as [number, number], []);
  const flyFrom = useMemo(() => [[LN_COL, COL_TOP, HZ] as [number, number, number]], []);
  const flyDelay = useMemo(() => (_c: number, r: number) => [0.33 + r * 0.006, 0.42 + r * 0.006] as [number, number], []);

  useStep(index, (sp, vis) => {
    // the action completes by 88% of the step; the camera glides to the next step after 86%
    const p = Math.min(1, sp / 0.88);
    const on = vis > 0.001;
    tok.current?.setOpacity(vis * seg(p, 0.06, 0.14) * (1 - seg(p, 0.38, 0.46)));
    xCol.current?.setOpacity(vis * seg(p, 0.06, 0.14) * (1 - 0.75 * seg(p, 0.18, 0.28)) * (1 - seg(p, 0.38, 0.46)));
    lnCol.current?.setProgress(p);
    lnCol.current?.setArc(0.5);
    lnCol.current?.setOpacity(on ? vis * seg(p, 0.18, 0.19) * (1 - 0.75 * seg(p, 0.33, 0.4)) * (1 - seg(p, 0.38, 0.46)) : 0);
    lnCol.current?.setBracketOpacity(vis * seg(p, 0.28, 0.32) * (1 - seg(p, 0.33, 0.4)));
    lnFly.current?.setProgress(p);
    lnFly.current?.setArc(0.4);
    lnFly.current?.setOpacity(on ? vis * (seg(p, 0.33, 0.34) - seg(p, 0.4, 0.45)) : 0);
    lnFly.current?.setBracketOpacity(0);

    const slabA = vis * seg(p, 0.2, 0.32);
    slab.current?.setOpacity(slabA);
    face.current?.setOpacity(slabA * 0.5);
    const sweep = easeInOut(seg(p, SWEEP[0], SWEEP[1]));
    const sweepOn = seg(p, SWEEP[0] - 0.02, SWEEP[0]) * (1 - seg(p, SWEEP[1], SWEEP[1] + 0.03));
    const bandY = WU.h / 2 - 0.25 - sweep * (WU.h - 0.5);
    slab.current?.setBand(bandY, 0.2, on ? 0.6 * sweepOn : 0);
    if (scan.current) {
      scan.current.position.y = WU.top - WU.h / 2 + bandY;
      (scan.current.material as THREE.LineBasicMaterial).opacity = 0.7 * vis * sweepOn;
      scan.current.visible = on && sweepOn > 0.01;
    }
    // the strip reveals in step with the scan (instances use the same p as their delay windows)
    const sweepP = SWEEP[0] + (SWEEP[1] - SWEEP[0]) * sweep;
    strip.current?.setProgress(p < SWEEP[0] ? 0 : p > SWEEP[1] ? 1 : sweepP);
    strip.current?.setOpacity(vis * seg(p, SWEEP[0] - 0.02, SWEEP[0]));
    topBars.current?.setProgress(p < SWEEP[0] ? 0 : p > SWEEP[1] ? 1 : sweepP);
    topBars.current?.setOpacity(vis * seg(p, SWEEP[0] - 0.02, SWEEP[0]));
    topBars.current?.setMix(seg(p, 0.74, 0.9));
    const labA = vis * seg(p, 0.74, 0.84);
    if (leaders.current) {
      (leaders.current.material as THREE.LineBasicMaterial).opacity = 0.35 * labA;
      leaders.current.visible = labA > 0.002;
    }
    values.current?.setOpacity(labA);
    topLayout.forEach((l, i) => names.current[i]?.setOpacity(vis * seg(p, 0.74 + l.rank * 0.012, 0.8 + l.rank * 0.012)));

    const lab = (i: number, a: number) => fade(labels.current[i], vis * a);
    lab(0, seg(p, 0.08, 0.16) * (1 - seg(p, 0.36, 0.44)));
    lab(1, seg(p, 0.24, 0.32) * (1 - seg(p, 0.38, 0.46)));
    lab(2, seg(p, 0.26, 0.34) * (1 - seg(p, 0.38, 0.46)));
    lab(3, seg(p, 0.22, 0.32));
    lab(4, seg(p, 0.26, 0.34));
    lab(5, seg(p, 0.44, 0.54));
    lab(6, seg(p, 0.5, 0.6));
    lab(7, seg(p, 0.5, 0.6));
    lab(8, seg(p, 0.86, 0.94));
  });

  const box = columnBox(X_COL, COL_TOP);
  return (
    <group>
      <TextPlane ref={tok} text={visibleToken(trace.tokens[t].text)} height={0.34} position={[X_COL, COL_TOP + 0.75, HZ]} />
      <Columns ref={xCol} data={trace.resid[12]} D={768} xs={xs} rows={rows} y0={COL_TOP} z={HZ} color={C.token} />
      <Columns ref={lnCol} data={trace.lnF} D={768} xs={lnXs} rows={rows} y0={COL_TOP} z={HZ} color={C.norm} from={lnFrom} delay={lnDelay} />
      <Columns ref={lnFly} data={trace.lnF} D={768} xs={[WU.x - WU.w / 2 - 0.42]} rows={rows} y0={WU.top - 0.25} z={HZ} color={C.norm} from={flyFrom} delay={flyDelay} />
      <Glass ref={slab} size={[WU.w, WU.h, WU.d]} position={[WU.x, WU.top - WU.h / 2, HZ]} color={C.token} fill={0.0025} rim={0.18} edge={0.42} />
      <GlyphField ref={face} items={faceItems} />
      <Lines ref={scan} points={[WU.x - WU.w / 2, 0, HZ + 0.2, STRIP.x0 + STRIP.len, 0, HZ + 0.2]} color={C.logit} opacity={0} />
      <Quads ref={strip} items={stripItems} />
      <Quads ref={topBars} items={topItems} renderOrder={9} />
      <Lines ref={leaders} points={leaderPts} color={C.dim} opacity={0} />
      <GlyphField ref={values} items={valueItems} />
      {topLayout.map((l, i) => (
        <TextPlane key={l.id} ref={(e) => void (names.current[i] = e)} text={visibleToken(trace.vocabText[l.id] ?? String(l.id))} height={NAME_H} position={[NAME_X, l.ly, HZ]} anchor="left" />
      ))}

      <Label ref={(e) => void (labels.current[0] = e)} position={[X_COL, box.y0 - 0.45, HZ]} tex={`x^{12}_T\\in\\mathbb{R}^{768}`} size={16} />
      <Label ref={(e) => void (labels.current[1] = e)} position={[LN_COL, box.y0 - 0.45, HZ]} tex={`\\text{LN}_f(x^{12}_T)`} size={16} color="#cfd5dd" />
      <Label ref={(e) => void (labels.current[2] = e)} position={[LN_COL, box.y0 - 0.95, HZ]} tex={`\\mu=${fmt(trace.lnFMean[t])},\\ \\sigma=${fmt(trace.lnFStd[t])}`} size={13} color="#9a9ea6" />
      <Label ref={(e) => void (labels.current[3] = e)} position={[WU.x, WU.top + 0.85, HZ]} tex="W_E^{\top}" size={30} />
      <Label ref={(e) => void (labels.current[4] = e)} position={[WU.x, WU.top + 0.3, HZ]} tex={`\\text{tied: the same table as }W_E`} size={12} color="#9a9ea6" />
      <Label ref={(e) => void (labels.current[5] = e)} position={[STRIP.x0 + STRIP.len / 2, WU.top + 0.85, HZ]} tex={`\\ell = \\text{LN}_f(x^{12}_T)\\,W_E^{\\top}\\in\\mathbb{R}^{50{,}257}`} size={17} />
      <Label ref={(e) => void (labels.current[6] = e)} position={[STRIP.x0, WU.top + 0.3, HZ]} tex={fmt(lo)} size={11} color="#8d929a" />
      <Label ref={(e) => void (labels.current[7] = e)} position={[STRIP.x0 + STRIP.len, WU.top + 0.3, HZ]} tex={fmt(hi)} size={11} color="#8d929a" />
      <Label ref={(e) => void (labels.current[8] = e)} position={[STRIP.x0 + STRIP.len / 2, WU.top - WU.h - 0.4, HZ]} tex={`50{,}257\\text{ logits, one per vocabulary row}`} size={13} color="#9a9ea6" />
    </group>
  );
}
