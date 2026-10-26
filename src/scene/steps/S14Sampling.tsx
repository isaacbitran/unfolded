// Step 14 — choosing a token. The 20 largest logits leave the strip and line up as a horizontal bar
// chart (sorted). Softmax turns them into probabilities; the temperature τ sweeps 1 → 0.5 → 1.5 →
// 0.6 with the real distribution recomputed over all 50,257 logits at every frame. Top-k (k = 12)
// cuts the tail, the cumulative sum Σp is drawn beside the chart and top-p (0.9) cuts where it
// crosses 0.9, the survivors are renormalised, and a marker rolls along the stacked bar of kept
// probabilities to a seeded uniform draw u, stopping on the sampled token.

import { useThree } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { distribution, rng, sample, topIndices, type SampleParams } from '../../model/sample';
import type { SceneTrace } from '../../model/trace';
import { Lines } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TextPlane, measureText, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { C, easeInOut, easeOut, fmt, lerp, seg } from '../theme';
import { useStep } from '../useStep';
import { Quads, rgba, type QuadItem, type QuadsHandle } from './S11MLP';
import { HZ, NAME_H, NAME_X, STRIP, logitRange, stripLen, topLabelLayout, vocabY } from './S13Unembed';

const N = 20;
const CX0 = 0.4; // bars start here
const ROW0 = 3.15;
const ROW_H = 0.34;
const PS = 9; // world units per unit of probability
const VAL_X = CX0 + 5.0;
const CUM = { x0: 6.75, w: 2.0 };
const SAMPLE_Y = ROW0 - N * ROW_H - 0.55;
const SW = 5.0; // sampling bar width (kept mass 1)
const BAR_H = 0.2;
const FINAL: SampleParams = { temperature: 0.6, topK: 12, topP: 0.9 };
const SEED = 24;

const rowY = (r: number) => ROW0 - r * ROW_H;

/** τ over the step: 1 → 0.5 → 1.5 → 0.6 */
function tauAt(p: number) {
  if (p < 0.3) return 1;
  if (p < 0.36) return lerp(1, 0.5, easeInOut(seg(p, 0.3, 0.36)));
  if (p < 0.38) return 0.5;
  if (p < 0.46) return lerp(0.5, 1.5, easeInOut(seg(p, 0.38, 0.46)));
  if (p < 0.48) return 1.5;
  return lerp(1.5, FINAL.temperature, easeInOut(seg(p, 0.48, 0.54)));
}

/** Plain-text tags in the shared label layer, positioned and rewritten imperatively each frame. */
function useTags(n: number, size: number, color: string) {
  const camera = useThree((s) => s.camera);
  const viewport = useThree((s) => s.size);
  const els = useMemo(
    () =>
      Array.from({ length: n }, () => {
        const d = document.createElement('div');
        d.className = 'label3d';
        d.style.fontSize = `${size}px`;
        d.style.color = color;
        d.style.visibility = 'hidden';
        return d;
      }),
    [n, size, color],
  );
  useEffect(() => {
    let layer = document.querySelector('.label-layer');
    if (!layer) {
      layer = document.createElement('div');
      layer.className = 'label-layer';
      document.body.appendChild(layer);
    }
    els.forEach((e) => layer!.appendChild(e));
    return () => els.forEach((e) => e.remove());
  }, [els]);
  const v = useMemo(() => new THREE.Vector3(), []);
  return useMemo(
    () => ({
      set(i: number, x: number, y: number, z: number, html: string, a: number, align: 'left' | 'right' | 'center' = 'left', col?: string) {
        const el = els[i];
        if (!el) return;
        if (a < 0.003) {
          if (el.style.visibility !== 'hidden') el.style.visibility = 'hidden';
          return;
        }
        if (el.innerHTML !== html) el.innerHTML = html;
        if (col && el.style.color !== col) el.style.color = col;
        el.style.visibility = 'visible';
        el.style.opacity = a.toFixed(3);
        v.set(x, y, z).project(camera);
        const px = (v.x * 0.5 + 0.5) * viewport.width;
        const py = (-v.y * 0.5 + 0.5) * viewport.height;
        const tx = align === 'left' ? 0 : align === 'right' ? -100 : -50;
        el.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px) translate(${tx}%, -50%)`;
      },
    }),
    [els, camera, viewport, v],
  );
}

const pct = (x: number) => (x >= 0.1 ? (100 * x).toFixed(1) : x >= 0.001 ? (100 * x).toFixed(2) : (100 * x).toFixed(3)) + '%';

/** The token the walkthrough samples in step 14 (step 15 appends this same token). */
export function walkthroughSample(trace: SceneTrace) {
  const { id } = sample(trace.logitsLast, FINAL, rng(SEED));
  return { id, text: trace.vocabText[id] ?? `#${id}` };
}

export function S14Sampling({ index, trace }: { index: number; trace: SceneTrace }) {
  const logits = trace.logitsLast;
  const { lo, hi } = useMemo(() => logitRange(logits), [logits]);
  const top = useMemo(() => topIndices(logits, N), [logits]);
  const s13 = useMemo(() => topLabelLayout(logits, top.slice(0, 10), lo, hi), [logits, top, lo, hi]);
  const final = useMemo(() => distribution(logits, FINAL, N), [logits]);
  const draw = useMemo(() => sample(logits, FINAL, rng(SEED)), [logits]);
  const kept = useMemo(() => final.filter((c) => c.kept > 0).length, [final]);
  const sampledRank = useMemo(() => final.findIndex((c) => c.id === draw.id), [final, draw]);

  // softmax over the whole vocabulary at temperature τ, for the 20 ranked ids (cached per 0.005 of τ)
  const cache = useRef(new Map<number, { prob: number[]; cum: number[] }>());
  const probsAt = (tau: number) => {
    const key = Math.round(tau * 200);
    const hit = cache.current.get(key);
    if (hit) return hit;
    const tt = key / 200;
    let z = 0;
    for (let i = 0; i < logits.length; i++) z += Math.exp((logits[i] - hi) / tt);
    const prob = top.map((id) => Math.exp((logits[id] - hi) / tt) / z);
    let c = 0;
    const cum = prob.map((x) => (c += x));
    const out = { prob, cum };
    cache.current.set(key, out);
    return out;
  };

  const bars = useRef<QuadsHandle>(null);
  const segs = useRef<QuadsHandle>(null);
  const marker = useRef<QuadsHandle>(null);
  const names = useRef<(TextPlaneHandle | null)[]>([]);
  const nameGroups = useRef<(THREE.Group | null)[]>([]);
  const cumLine = useRef<THREE.Line>(null);
  const cumDots = useRef<QuadsHandle>(null);
  const frame = useRef<THREE.LineSegments>(null);
  const kCut = useRef<THREE.LineSegments>(null);
  const pCutLine = useRef<THREE.LineSegments>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const vals = useTags(N, 12, '#a9adb5');
  const tags = useTags(4, 15, '#f4eee2');

  const nameW = useMemo(() => top.map((id) => measureText(visibleToken(trace.vocabText[id] ?? String(id)), NAME_H)), [top, trace]);
  // start: rank r's bar and label where step 13 left them
  const starts = useMemo(
    () =>
      top.map((id, r) => {
        const l = s13.find((q) => q.id === id);
        return { y: vocabY(id), len: stripLen(logits[id], lo, hi), ly: l ? l.ly : vocabY(id), lx: l ? NAME_X : NAME_X + 0.6, labelled: !!l, r };
      }),
    [top, s13, logits, lo, hi],
  );

  const barItems = useMemo(() => top.map((): QuadItem => ({ x: 0, y: 0, z: HZ, w: 0, h: BAR_H, a: rgba(C.logit, 1) })), [top]);
  const dotItems = useMemo(() => top.map((): QuadItem => ({ x: 0, y: 0, z: HZ + 0.01, w: 0.09, h: 0.09, a: rgba(C.logit, 1) })), [top]);
  const markerItems = useMemo((): QuadItem[] => [{ x: CX0, y: SAMPLE_Y, z: HZ + 0.05, w: 0.26, h: 0.26, a: rgba(C.logit, 2.2) }], []);
  // stacked bar of the kept (renormalised) probabilities, in rank order
  const segLayout = useMemo(() => {
    let x = CX0;
    return final.filter((c) => c.kept > 0).map((c, r) => {
      const w = c.kept * SW;
      const s = { x0: x, w, r, id: c.id };
      x += w;
      return s;
    });
  }, [final]);
  const segItems = useMemo(
    () => segLayout.map((s): QuadItem => ({ x: s.x0 + s.w / 2, y: SAMPLE_Y, z: HZ, w: Math.max(0.01, s.w - 0.03), h: 0.16, from: [s.x0, SAMPLE_Y, HZ], fromSize: [0, 0.16], a: rgba(C.logit, s.r % 2 ? 0.35 : 0.6), b: rgba(C.logit, s.r === sampledRank ? 1.8 : s.r % 2 ? 0.25 : 0.4), delay: [0.8 + s.r * 0.006, 0.84 + s.r * 0.006], key: 0.5 })),
    [segLayout, sampledRank],
  );
  const cumGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    return g;
  }, []);
  const cumMat = useMemo(() => new THREE.LineBasicMaterial({ color: C.logit, transparent: true, opacity: 0, depthWrite: false }), []);
  const cumObj = useMemo(() => new THREE.Line(cumGeo, cumMat), [cumGeo, cumMat]);
  const yTop = rowY(0) + 0.3;
  const yBot = rowY(N - 1) - 0.3;
  const framePts = useMemo(
    () => [CUM.x0, yTop, HZ, CUM.x0, yBot, HZ, CUM.x0 + CUM.w, yTop, HZ, CUM.x0 + CUM.w, yBot, HZ],
    [yTop, yBot],
  );
  const pX = CUM.x0 + FINAL.topP * CUM.w;
  const kY = (rowY(FINAL.topK - 1) + rowY(FINAL.topK)) / 2;
  const pY = (rowY(kept - 1) + rowY(kept)) / 2;
  const kPts = useMemo(() => [CX0 - 2.2, kY, HZ, CX0 + 0.9, kY, HZ, CX0 + 2.5, kY, HZ, CUM.x0 + CUM.w, kY, HZ], [kY]);
  const pPts = useMemo(() => [CX0 - 2.2, pY, HZ, CX0 + 0.9, pY, HZ, CX0 + 3.6, pY, HZ, CUM.x0 + CUM.w, pY, HZ, pX, yTop, HZ, pX, yBot, HZ], [pY, pX, yTop, yBot]);

  const lineOp = (o: THREE.LineSegments | THREE.Line | null, a: number) => {
    if (!o) return;
    (o.material as THREE.LineBasicMaterial).opacity = a;
    o.visible = a > 0.002;
  };

  useStep(index, (sp, vis) => {
    // the action completes by 88% of the step; the camera glides to the next step after 86%
    const p = Math.min(1, sp / 0.88);
    const on = vis > 0.001;
    if (!on) {
      for (let i = 0; i < N; i++) vals.set(i, 0, 0, 0, '', 0);
      for (let i = 0; i < 4; i++) tags.set(i, 0, 0, 0, '', 0);
    }
    const tau = tauAt(p);
    const soft = easeInOut(seg(p, 0.18, 0.28));
    const { prob, cum } = probsAt(tau);
    const kCutA = seg(p, 0.56, 0.62);
    const pCutA = seg(p, 0.7, 0.75);
    const renorm = easeInOut(seg(p, 0.76, 0.82));
    const glow = seg(p, 0.93, 0.97);

    for (let r = 0; r < N; r++) {
      const st = starts[r];
      const m = easeInOut(seg(p, 0.03 + r * 0.003, 0.12 + r * 0.003));
      const y = lerp(st.y, rowY(r), m);
      const x0 = lerp(STRIP.x0, CX0, m);
      const logitLen = st.len;
      const fin = final[r];
      const probLen = lerp(prob[r] * PS, fin.kept * PS, renorm * (fin.kept > 0 ? 1 : 0));
      const len = Math.max(0.004, lerp(logitLen, probLen, soft));
      const dropped = (r >= FINAL.topK ? kCutA : 0) + (r < FINAL.topK && fin.kept === 0 ? pCutA : 0);
      const isSampled = r === sampledRank;
      const k = (0.3 + 0.55 * Math.min(1, len / 4.5) ** 2) * (1 - 0.8 * dropped) + (isSampled ? 0.75 * glow : 0);
      bars.current?.setRect(r, x0 + len / 2, y, len, lerp(0.035, BAR_H, m));
      const c = rgba(C.logit, k);
      bars.current?.setColor(r, c[0], c[1], c[2], 1);
      // names: from step 13's label spot (left-anchored) to the right-aligned spot left of the bar
      const g = nameGroups.current[r];
      const tx = CX0 - 0.25 - nameW[r];
      if (g) g.position.set(lerp(st.lx, tx, m), lerp(st.ly, rowY(r), m), HZ);
      const nameA = st.labelled ? 1 : seg(p, 0.1, 0.18);
      names.current[r]?.setOpacity(vis * nameA * (1 - 0.75 * dropped) * (isSampled ? 1 : 1 - 0.3 * glow));
      // value tags: logit, then probability, then kept probability
      let html: string;
      if (soft < 0.5) html = fmt(logits[top[r]]);
      else if (renorm < 0.5) html = pct(prob[r]);
      else html = fin.kept > 0 ? pct(fin.kept) : '0%';
      const vA = vis * seg(p, 0.14, 0.2) * (1 - 0.6 * Math.sin(Math.PI * seg(p, 0.21, 0.25))) * (1 - 0.65 * dropped);
      vals.set(r, VAL_X, rowY(r), HZ, html, vA, 'left');
      // cumulative dots
      cumDots.current?.setRect(r, CUM.x0 + cum[r] * CUM.w, rowY(r), 0.08, 0.08);
      const crossed = r > 0 && cum[r - 1] >= FINAL.topP;
      const cd = rgba(C.logit, crossed ? 0.35 : 1.0);
      cumDots.current?.setColor(r, cd[0], cd[1], cd[2], 1);
      const pos = cumGeo.getAttribute('position') as THREE.BufferAttribute;
      pos.setXYZ(r, CUM.x0 + cum[r] * CUM.w, rowY(r), HZ);
    }
    bars.current?.commit();
    cumDots.current?.commit();
    (cumGeo.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    bars.current?.setOpacity(vis);

    // cumulative column: frame, line drawn row by row, threshold
    const cumA = vis * seg(p, 0.62, 0.66);
    const drawn = seg(p, 0.64, 0.72);
    cumGeo.setDrawRange(0, Math.max(0, Math.round(N * drawn)));
    lineOp(cumObj, 0.8 * cumA);
    cumDots.current?.setOpacity(cumA);
    cumDots.current?.setProgress(1);
    lineOp(frame.current, 0.3 * cumA);
    lineOp(kCut.current, 0.55 * vis * kCutA * (1 - 0.6 * seg(p, 0.72, 0.78)));
    lineOp(pCutLine.current, 0.7 * vis * seg(p, 0.66, 0.7));

    // sampling bar and marker
    segs.current?.setProgress(p);
    segs.current?.setOpacity(vis * seg(p, 0.79, 0.8));
    segs.current?.setMix(glow);
    const roll = easeOut(seg(p, 0.84, 0.93));
    const mx = CX0 + draw.u * SW * roll;
    marker.current?.setRect(0, mx, SAMPLE_Y, 0.26, 0.26);
    marker.current?.commit();
    marker.current?.setOpacity(vis * seg(p, 0.83, 0.85));

    const lab = (i: number, a: number) => fade(labels.current[i], vis * a);
    lab(0, seg(p, 0.12, 0.18) * (1 - seg(p, 0.19, 0.23)));
    lab(1, seg(p, 0.22, 0.28));
    lab(2, kCutA);
    lab(3, seg(p, 0.66, 0.7));
    lab(4, seg(p, 0.62, 0.66));
    lab(5, seg(p, 0.79, 0.83));
    lab(6, glow);
    lab(7, seg(p, 0.62, 0.66));
    // τ readout and the final parameters
    const tauA = vis * seg(p, 0.24, 0.3);
    tags.set(0, CX0 + 4.1, ROW0 + 0.62, HZ, `<i>τ</i> = ${tau.toFixed(2)}`, tauA, 'left');
    tags.set(1, CX0 - 0.25, SAMPLE_Y, HZ, `<i>u</i> = ${draw.u.toFixed(3)}`, vis * seg(p, 0.84, 0.88), 'right', '#a9adb5');
    tags.set(2, 0, 0, HZ, '', 0);
    tags.set(3, 0, 0, HZ, '', 0);
  });

  const sampledText = visibleToken(trace.vocabText[draw.id] ?? String(draw.id));
  return (
    <group>
      <Quads ref={bars} items={barItems} renderOrder={9} />
      {top.map((id, r) => (
        <group key={id} ref={(g) => void (nameGroups.current[r] = g)}>
          <TextPlane ref={(e) => void (names.current[r] = e)} text={visibleToken(trace.vocabText[id] ?? String(id))} height={NAME_H} anchor="left" />
        </group>
      ))}
      <Lines ref={frame} points={framePts} color={C.dim} opacity={0} />
      <primitive object={cumObj} ref={cumLine} renderOrder={6} />
      <Quads ref={cumDots} items={dotItems} round renderOrder={9} />
      <Lines ref={kCut} points={kPts} color={C.logit} opacity={0} />
      <Lines ref={pCutLine} points={pPts} color={C.logit} opacity={0} />
      <Quads ref={segs} items={segItems} />
      <Quads ref={marker} items={markerItems} round renderOrder={11} />

      <Label ref={(e) => void (labels.current[0] = e)} position={[CX0 + 2.2, ROW0 + 0.62, HZ]} tex={`\\ell_v\\ \\text{(top 20 of }50{,}257)`} size={17} />
      <Label ref={(e) => void (labels.current[1] = e)} position={[CX0 + 3.8, ROW0 + 0.62, HZ]} tex={`p_v=\\dfrac{e^{\\ell_v/\\tau}}{\\sum_{u} e^{\\ell_u/\\tau}}`} size={15} align="right" />
      <Label ref={(e) => void (labels.current[2] = e)} position={[CX0 + 1.0, kY, HZ]} tex={`\\text{top-}k = ${FINAL.topK}`} size={14} align="left" color="#d8d4ca" />
      <Label ref={(e) => void (labels.current[3] = e)} position={[CX0 + 1.0, pY, HZ]} tex={`\\text{top-}p = ${FINAL.topP}\\text{: keeps } ${kept}`} size={14} align="left" color="#d8d4ca" />
      <Label ref={(e) => void (labels.current[4] = e)} position={[CUM.x0 + CUM.w / 2, ROW0 + 0.62, HZ]} tex={`\\textstyle\\sum p`} size={15} />
      <Label ref={(e) => void (labels.current[5] = e)} position={[CX0 + SW / 2, SAMPLE_Y - 0.42, HZ]} tex={`\\text{kept, renormalised: }${kept}\\text{ tokens, }\\textstyle\\sum = 1`} size={12} color="#9a9ea6" />
      <Label ref={(e) => void (labels.current[6] = e)} position={[CX0 + SW + 0.3, SAMPLE_Y, HZ]} text={`→ ${sampledText}`} size={18} align="left" />
      <Label ref={(e) => void (labels.current[7] = e)} position={[pX, yTop + 0.02, HZ]} tex="0.9" size={11} color="#8d929a" />
    </group>
  );
}
