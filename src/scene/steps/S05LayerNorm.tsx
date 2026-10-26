// Step 5 — LayerNorm on block 1's input, for the last token. All 768 entries of x = resid[0][T−1]
// become hairline ticks on a number line. The strip is centered (subtract μ; a loupe shows the tiny
// real shift near 0), rescaled to unit standard deviation (divide by σ), multiplied per dimension by
// γ, the axis zooms in, and β shifts each tick. A bracketed column of numbers updates in sync.
// At the end the normalized column returns to its place beside the other five normalized tokens.

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { Columns, type ColumnsHandle } from '../primitives/Columns';
import { Lines } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TextPlane, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { TokenHeader, type TokenHeaderHandle } from '../primitives/TokenHeader';
import { C, L, colX, easeInOut, lerp, seg } from '../theme';
import { useStep } from '../useStep';
import { columnBox } from '../vectors';
import { COL_TOP, TOKEN_ID_Y, TOKEN_LABEL_Y } from './S02Embedding';

const D = 768;
/** Column layout (world units). */
const ROW_Y0 = 3.0;
const X_IN = -4.4;
const X_STAGE = -1.9;
const X_GAMMA = 1.0;
const X_BETA = 3.2;
const XS_STAGE = [X_STAGE];
const XS_GAMMA = [X_GAMMA];
const XS_BETA = [X_BETA];
/** Number line. */
const AX = { cx: 1.4, y: -1.75, half: 4.6 };
const S0 = AX.half / 7.2; // world units per value unit while the raw/normalized strip is shown
const S1 = AX.half / 0.48; // after the zoom that reveals γ ⊙ x̂ + β
/** Loupe under the axis: a ±0.006 window around 0, enlarged. */
const LOUPE = { y: -3.25, half: 1.3, range: 0.0062 };

/** Phase windows in step progress p. */
const PH = {
  focus: [0.03, 0.12] as const, // other columns fade, the last one moves left
  emit: [0.1, 0.2] as const, // ticks fly from the column onto the number line
  mu: [0.19, 0.23] as const, // mean marker appears
  center: [0.25, 0.35] as const, // subtract μ
  sigma: [0.37, 0.41] as const, // σ band appears
  scale: [0.43, 0.56] as const, // divide by σ
  gamma: [0.6, 0.69] as const, // multiply by γ
  zoom: [0.7, 0.77] as const, // axis zooms in
  beta: [0.79, 0.86] as const, // add β
  back: [0.88, 0.96] as const, // return to the six-column layout
};

export function S05LayerNorm({ index, trace }: { index: number; trace: SceneTrace }) {
  const T = trace.T;
  const k = T - 1;
  const L0 = trace.layers[0];
  const mu = L0.ln1Mean[k];
  const sd = L0.ln1Std[k];
  const xs = useMemo(() => trace.tokens.map((_, i) => colX(i, T)), [trace, T]);

  // the four states of the focused vector, each a [1, 768] tensor
  const states = useMemo(() => {
    const x = trace.resid[0].slice(k * D, (k + 1) * D);
    const c = x.map((v) => v - mu);
    const h = c.map((v) => v / sd);
    const y = L0.ln1.slice(k * D, (k + 1) * D);
    return { x, c, h, y };
  }, [trace, k, mu, sd, L0]);

  const others = useRef<ColumnsHandle>(null);
  const othersLn = useRef<ColumnsHandle>(null);
  const colIn = useRef<ColumnsHandle>(null);
  const colC = useRef<ColumnsHandle>(null);
  const colH = useRef<ColumnsHandle>(null);
  const colY = useRef<ColumnsHandle>(null);
  const colG = useRef<ColumnsHandle>(null);
  const colB = useRef<ColumnsHandle>(null);
  const inGroup = useRef<THREE.Group>(null);
  const yGroup = useRef<THREE.Group>(null);
  const header = useRef<TokenHeaderHandle>(null);
  const tokPlane = useRef<TextPlaneHandle>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const axisLabelGroups = useRef<(THREE.Group | null)[]>([]);
  const axisLabels = useRef<(HTMLDivElement | null)[]>([]);
  const axisLine = useRef<THREE.LineSegments>(null);
  const loupeLines = useRef<THREE.LineSegments>(null);
  const muGroup = useRef<THREE.Group>(null);
  const loupeMuGroup = useRef<THREE.Group>(null);

  // ---------- dynamic line geometry: 768 ticks, featured ticks, axis marks, σ band, loupe ticks ----------
  const featured = useMemo(() => [0, 1, 2, 3, 4, 5, D - 1], []);
  const nearZero = useMemo(() => {
    const out: number[] = [];
    for (let i = 0; i < D; i++) if (Math.abs(states.x[i]) < LOUPE.range * 1.6) out.push(i);
    return out;
  }, [states]);
  const dyn = useMemo(() => {
    const mk = (nSeg: number, color: string, opacity: number, additive = true) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nSeg * 6), 3));
      const m = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
        blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
      });
      const obj = new THREE.LineSegments(g, m);
      obj.frustumCulled = false;
      obj.renderOrder = 6;
      return obj;
    };
    return {
      ticks: mk(D, '#b9b4aa', 0.3, false),
      feat: mk(featured.length, '#ffffff', 0.95, false),
      marks: mk(16, C.dim, 0.7, false),
      sigma: mk(3, C.norm, 0.9, false),
      loupe: mk(nearZero.length, '#b9b4aa', 0.75, false),
    };
  }, [featured, nearZero]);

  // ---------- side view: the same two operations on a 3-dimensional vector ----------
  const geo = useMemo(() => makeGeoInset(), []);
  const geoLabels = useRef<(HTMLDivElement | null)[]>([]);

  const AXIS_A = [-6, -4, -2, 0, 2, 4, 6];
  const AXIS_B = [-0.4, -0.2, 0, 0.2, 0.4];

  useStep(index, (p, vis) => {
    const on = vis > 0.001;
    // ---- columns ----
    const focus = easeInOut(seg(p, PH.focus[0], PH.focus[1]));
    const back = easeInOut(seg(p, PH.back[0], PH.back[1]));
    header.current?.setOpacity(vis * (1 - focus));
    header.current?.setIdOpacity(0);
    others.current?.setOpacity(vis * (1 - focus));
    othersLn.current?.setOpacity(vis * back);
    // the focused input column glides from its slot to the left
    if (inGroup.current) {
      inGroup.current.position.set(lerp(colX(k, T), X_IN, focus), lerp(COL_TOP, ROW_Y0, focus), 0);
    }
    colIn.current?.setOpacity(vis * (1 - back));
    tokPlane.current?.setOpacity(vis * focus * (1 - back));
    // stage column: x − μ → x̂ → LN(x), each crossfading as its operation happens
    const cIn = seg(p, PH.center[1] - 0.035, PH.center[1]);
    const cScale = seg(p, PH.scale[1] - 0.03, PH.scale[1]);
    const cOut = seg(p, PH.beta[1] - 0.03, PH.beta[1]);
    colC.current?.setOpacity(vis * cIn * (1 - cScale));
    colH.current?.setOpacity(vis * cScale * (1 - cOut));
    colY.current?.setOpacity(vis * cOut);
    if (yGroup.current) yGroup.current.position.set(lerp(0, colX(k, T) - X_STAGE, back), lerp(0, COL_TOP - ROW_Y0, back), 0);
    const gb = seg(p, PH.gamma[0] - 0.04, PH.gamma[0]) * (1 - back);
    colG.current?.setOpacity(vis * gb);
    colB.current?.setOpacity(vis * seg(p, PH.beta[0] - 0.04, PH.beta[0]) * (1 - back));

    // ---- strip ----
    const emit = seg(p, PH.emit[0], PH.emit[1]);
    const t1 = easeInOut(seg(p, PH.center[0], PH.center[1]));
    const t2 = easeInOut(seg(p, PH.scale[0], PH.scale[1]));
    const t3 = easeInOut(seg(p, PH.gamma[0], PH.gamma[1]));
    const tz = easeInOut(seg(p, PH.zoom[0], PH.zoom[1]));
    const t4 = easeInOut(seg(p, PH.beta[0], PH.beta[1]));
    const s = S0 * Math.pow(S1 / S0, tz); // geometric zoom
    const stripA = vis * seg(p, PH.emit[0], PH.emit[0] + 0.03) * (1 - seg(p, PH.back[0], PH.back[0] + 0.05));
    const value = (i: number) => {
      const x = states.x[i];
      const c = x - mu * t1;
      const h = c * lerp(1, 1 / sd, t2);
      return h * lerp(1, trace.ln1Gamma[i], t3) + trace.ln1Beta[i] * t4;
    };
    const box = columnBox(X_IN, ROW_Y0);
    const tickH = 0.32;
    const pos = dyn.ticks.geometry.getAttribute('position') as THREE.BufferAttribute;
    const arr = pos.array as Float32Array;
    for (let i = 0; i < D; i++) {
      // stagger by index so the strip pours out of the column top to bottom
      const a = easeInOut(seg(emit, (i / D) * 0.55, (i / D) * 0.55 + 0.45));
      const fx = X_IN + 0.3, fy = lerp(box.y1 - 0.2, box.y0 + 0.2, i / D);
      const tx = AX.cx + value(i) * s;
      const x = lerp(fx, tx, a);
      const y = lerp(fy, AX.y, a);
      const h = lerp(0.04, tickH, a);
      const o = i * 6;
      arr[o] = x; arr[o + 1] = y - h / 2; arr[o + 2] = 0;
      arr[o + 3] = x; arr[o + 4] = y + h / 2; arr[o + 5] = 0;
    }
    pos.needsUpdate = true;
    (dyn.ticks.material as THREE.LineBasicMaterial).opacity = 0.42 * stripA;
    dyn.ticks.visible = on && stripA > 0.002;

    // featured ticks: the entries shown in the column, taller and bright, born at their row
    const fpos = dyn.feat.geometry.getAttribute('position') as THREE.BufferAttribute;
    const farr = fpos.array as Float32Array;
    featured.forEach((i, r) => {
      const row = r < 6 ? r : 7;
      const a = easeInOut(seg(emit, 0.1 + r * 0.05, 0.6 + r * 0.05));
      const fx = X_IN + 0.75, fy = ROW_Y0 - row * L.rowH;
      const x = lerp(fx, AX.cx + value(i) * s, a);
      const y = lerp(fy, AX.y + 0.06, a);
      const h = lerp(0.0, 0.5, a);
      const o = r * 6;
      farr[o] = x; farr[o + 1] = y - h / 2; farr[o + 2] = 0;
      farr[o + 3] = x; farr[o + 4] = y + h / 2; farr[o + 5] = 0;
    });
    fpos.needsUpdate = true;
    (dyn.feat.material as THREE.LineBasicMaterial).opacity = 0.9 * stripA * seg(emit, 0.15, 0.4);
    dyn.feat.visible = on && stripA > 0.002;

    // axis line + marks at the labelled values
    if (axisLine.current) {
      (axisLine.current.material as THREE.LineBasicMaterial).opacity = 0.45 * stripA;
      axisLine.current.visible = on && stripA > 0.002;
    }
    const mpos = dyn.marks.geometry.getAttribute('position') as THREE.BufferAttribute;
    const marr = mpos.array as Float32Array;
    marr.fill(0);
    const fadeA = 1 - seg(tz, 0.2, 0.6);
    const fadeB = seg(tz, 0.45, 0.9);
    const placeMark = (slot: number, v: number, a: number) => {
      const x = AX.cx + v * s;
      const inside = Math.abs(x - AX.cx) <= AX.half + 1e-3;
      const o = slot * 6;
      if (inside && a > 0.01) {
        marr[o] = x; marr[o + 1] = AX.y - 0.42; marr[o + 2] = 0;
        marr[o + 3] = x; marr[o + 4] = AX.y - 0.3; marr[o + 5] = 0;
      }
      const g = axisLabelGroups.current[slot];
      if (g) g.position.x = x;
      // fade labels near the ends so nothing pops at the edge
      const edge = 1 - seg(Math.abs(x - AX.cx), AX.half - 0.3, AX.half + 0.05);
      fade(axisLabels.current[slot], stripA * a * edge);
    };
    AXIS_A.forEach((v, j) => placeMark(j, v, fadeA));
    AXIS_B.forEach((v, j) => placeMark(AXIS_A.length + j, v, fadeB));
    mpos.needsUpdate = true;
    (dyn.marks.material as THREE.LineBasicMaterial).opacity = 0.75 * stripA;
    dyn.marks.visible = on && stripA > 0.002;

    // mean marker: a short silver line above the axis at μ, sliding to 0 while centering
    const muA = vis * seg(p, PH.mu[0], PH.mu[1]) * (1 - seg(p, PH.center[1] + 0.005, PH.sigma[0]));
    if (muGroup.current) {
      muGroup.current.position.x = AX.cx + mu * (1 - t1) * s;
      muGroup.current.visible = on && muA > 0.002;
      muGroup.current.children.forEach((c) => {
        const m = (c as THREE.LineSegments).material as THREE.LineBasicMaterial | undefined;
        if (m && 'opacity' in m) m.opacity = 0.95 * muA;
      });
    }
    fade(labels.current[0], muA);

    // loupe: ticks within ±0.006 of zero, enlarged; μ slides onto 0
    const loupeA = vis * seg(p, PH.mu[0] + 0.01, PH.mu[1] + 0.02) * (1 - seg(p, PH.center[1] + 0.03, PH.sigma[1]));
    const ls = LOUPE.half / LOUPE.range;
    const lpos = dyn.loupe.geometry.getAttribute('position') as THREE.BufferAttribute;
    const larr = lpos.array as Float32Array;
    larr.fill(0);
    nearZero.forEach((i, j) => {
      const v = states.x[i] - mu * t1;
      if (Math.abs(v) > LOUPE.range) return;
      const x = AX.cx + v * ls;
      const o = j * 6;
      larr[o] = x; larr[o + 1] = LOUPE.y - 0.16; larr[o + 2] = 0;
      larr[o + 3] = x; larr[o + 4] = LOUPE.y + 0.16; larr[o + 5] = 0;
    });
    lpos.needsUpdate = true;
    (dyn.loupe.material as THREE.LineBasicMaterial).opacity = 0.8 * loupeA;
    dyn.loupe.visible = on && loupeA > 0.002;
    if (loupeLines.current) {
      (loupeLines.current.material as THREE.LineBasicMaterial).opacity = 0.35 * loupeA;
      loupeLines.current.visible = on && loupeA > 0.002;
    }
    if (loupeMuGroup.current) {
      loupeMuGroup.current.position.x = AX.cx + mu * (1 - t1) * ls;
      loupeMuGroup.current.visible = on && loupeA > 0.002;
      const m = (loupeMuGroup.current.children[0] as THREE.LineSegments | undefined)?.material as THREE.LineBasicMaterial | undefined;
      if (m) m.opacity = 0.95 * loupeA;
    }
    fade(labels.current[1], loupeA);
    fade(labels.current[2], loupeA);
    fade(labels.current[3], loupeA);

    // σ band: brackets at ±σ (after centering), growing with the rescale to ±1
    const sigA = vis * seg(p, PH.sigma[0] + 0.005, PH.sigma[1]) * (1 - seg(p, PH.gamma[0] - 0.03, PH.gamma[0] + 0.01));
    const half = sd * lerp(1, 1 / sd, t2) * s;
    const spos = dyn.sigma.geometry.getAttribute('position') as THREE.BufferAttribute;
    const sarr = spos.array as Float32Array;
    const sy = AX.y + 0.5;
    const segs = [
      [AX.cx - half, sy, AX.cx + half, sy],
      [AX.cx - half, sy - 0.1, AX.cx - half, sy + 0.1],
      [AX.cx + half, sy - 0.1, AX.cx + half, sy + 0.1],
    ];
    segs.forEach((q, j) => sarr.set([q[0], q[1], 0, q[2], q[3], 0], j * 6));
    spos.needsUpdate = true;
    (dyn.sigma.material as THREE.LineBasicMaterial).opacity = 0.85 * sigA;
    dyn.sigma.visible = on && sigA > 0.002;
    fade(labels.current[4], sigA * (1 - seg(p, PH.scale[0], PH.scale[0] + 0.03)));
    fade(labels.current[5], sigA * seg(p, PH.scale[1] - 0.03, PH.scale[1]));

    // column captions
    const colLab = vis * seg(p, PH.focus[1] - 0.03, PH.focus[1]) * (1 - back);
    fade(labels.current[6], colLab); // x
    fade(labels.current[7], vis * cIn * (1 - cScale)); // x − μ
    fade(labels.current[8], vis * cScale * (1 - cOut)); // x̂
    fade(labels.current[9], vis * cOut * (1 - back)); // LN(x)
    fade(labels.current[10], vis * gb); // γ
    fade(labels.current[11], vis * seg(p, PH.beta[0] - 0.04, PH.beta[0]) * (1 - back)); // β
    // shapes under the columns
    fade(labels.current[12], colLab);
    fade(labels.current[13], vis * cIn * (1 - back));
    fade(labels.current[14], vis * gb);
    fade(labels.current[15], vis * seg(p, PH.beta[0] - 0.04, PH.beta[0]) * (1 - back));
    // formulas at the bottom: one per stage
    const fA = vis * seg(p, PH.mu[0], PH.mu[1]) * (1 - seg(p, PH.sigma[0] - 0.02, PH.sigma[0]));
    const fB = vis * seg(p, PH.sigma[0], PH.sigma[1]) * (1 - seg(p, PH.gamma[0] - 0.04, PH.gamma[0]));
    const fC = vis * seg(p, PH.gamma[0], PH.gamma[0] + 0.03) * (1 - seg(p, PH.back[0], PH.back[0] + 0.04));
    fade(labels.current[17], fA);
    fade(labels.current[18], fB);
    fade(labels.current[19], fC);
    fade(labels.current[20], vis * seg(p, PH.back[1] - 0.04, PH.back[1]));

    // side view: visible while centering and rescaling happen, gone before γ and β need the space
    const geoA = vis * seg(p, PH.mu[0], PH.mu[1] + 0.02) * (1 - seg(p, PH.scale[1] + 0.01, PH.gamma[0] - 0.02));
    geo.update(t1, t2, geoA, seg(p, PH.mu[0], PH.gamma[0]));
    geoLabels.current.forEach((el) => fade(el, geoA));
  });

  const ticksObj = dyn.ticks;
  const muLine = useMemo(() => [0, AX.y + 0.2, 0, 0, AX.y + 0.75, 0], []);
  const loupeMuLine = useMemo(() => [0, LOUPE.y - 0.3, 0, 0, LOUPE.y + 0.32, 0], []);
  const axisPts = useMemo(() => [AX.cx - AX.half - 0.15, AX.y, 0, AX.cx + AX.half + 0.15, AX.y, 0], []);
  // loupe frame: its own axis plus two faint lines from the main axis's zero to the loupe ends
  const loupePts = useMemo(
    () => [
      AX.cx - LOUPE.half, LOUPE.y, 0, AX.cx + LOUPE.half, LOUPE.y, 0,
      AX.cx, AX.y - 0.18, 0, AX.cx - LOUPE.half, LOUPE.y + 0.24, 0,
      AX.cx, AX.y - 0.18, 0, AX.cx + LOUPE.half, LOUPE.y + 0.24, 0,
      AX.cx, LOUPE.y - 0.08, 0, AX.cx, LOUPE.y - 0.22, 0,
    ],
    [],
  );
  const fmt4 = (v: number) => (v < 0 ? '-' : '') + Math.abs(v).toFixed(4);
  const capY = columnBox(0, ROW_Y0).y0 - 0.32;
  const xsOthers = useMemo(() => xs.slice(0, k), [xs, k]);
  const rowsOthers = useMemo(() => xsOthers.map((_, i) => i), [xsOthers]);
  const inXs = useMemo(() => [0], []);
  const kRows = useMemo(() => [k], [k]);
  const zeroRow = useMemo(() => [0], []);

  return (
    <group>
      <group position={GEO.pos}>
        <group ref={geo.setRoot}>
          <primitive object={geo.root} />
          <Label ref={(e) => void (geoLabels.current[0] = e)} position={geo.diagTip} tex="(1,\dots,1)" size={11} color="#8d929a" align="left" />
          <Label ref={(e) => void (geoLabels.current[1] = e)} position={geo.sphereTag} tex="\sqrt{d}" size={12} color={C.norm} align="left" />
        </group>
        <Label ref={(e) => void (geoLabels.current[2] = e)} position={[0, -1.5, 0]} tex="\text{the same two steps for } d = 3" size={12} color="#8d929a" />
      </group>
      <TokenHeader ref={header} trace={trace} y={TOKEN_LABEL_Y} idY={TOKEN_ID_Y} />
      <Columns ref={others} data={trace.resid[0]} D={D} xs={xsOthers} rows={rowsOthers} y0={COL_TOP} color={C.token} />
      <Columns ref={othersLn} data={L0.ln1} D={D} xs={xsOthers} rows={rowsOthers} y0={COL_TOP} color={C.norm} />
      <group ref={inGroup}>
        <Columns ref={colIn} data={trace.resid[0]} D={D} xs={inXs} rows={kRows} y0={0} color={C.token} />
        <TextPlane ref={tokPlane} text={visibleToken(trace.tokens[k].text)} height={0.36} position={[0, 0.85, 0]} />
      </group>
      <Columns ref={colC} data={states.c} D={D} xs={XS_STAGE} rows={zeroRow} y0={ROW_Y0} color={C.norm} />
      <Columns ref={colH} data={states.h} D={D} xs={XS_STAGE} rows={zeroRow} y0={ROW_Y0} color={C.norm} k={1.15} />
      <group ref={yGroup}>
        <Columns ref={colY} data={states.y} D={D} xs={XS_STAGE} rows={zeroRow} y0={ROW_Y0} color={C.norm} k={1.2} />
      </group>
      <Columns ref={colG} data={trace.ln1Gamma} D={D} xs={XS_GAMMA} rows={zeroRow} y0={ROW_Y0} color={C.norm} k={0.85} />
      <Columns ref={colB} data={trace.ln1Beta} D={D} xs={XS_BETA} rows={zeroRow} y0={ROW_Y0} color={C.norm} k={0.85} />

      <primitive object={ticksObj} />
      <primitive object={dyn.feat} />
      <primitive object={dyn.marks} />
      <primitive object={dyn.sigma} />
      <primitive object={dyn.loupe} />
      <Lines ref={axisLine} points={axisPts} color={C.dim} opacity={0} />
      <Lines ref={loupeLines} points={loupePts} color={C.dim} opacity={0} />
      <group ref={muGroup}>
        <Lines points={muLine} color={C.norm} opacity={0} />
        <Label ref={(e) => void (labels.current[0] = e)} position={[0, AX.y + 1.0, 0]} tex={`\\mu = ${fmt4(mu)}`} size={15} color={C.norm} />
      </group>
      <group ref={loupeMuGroup}>
        <Lines points={loupeMuLine} color={C.norm} opacity={0} />
      </group>
      <Label ref={(e) => void (labels.current[1] = e)} position={[AX.cx - LOUPE.half, LOUPE.y - 0.45, 0]} tex="-0.006" size={11} color="#8d929a" />
      <Label ref={(e) => void (labels.current[2] = e)} position={[AX.cx, LOUPE.y - 0.45, 0]} tex="0" size={11} color="#8d929a" />
      <Label ref={(e) => void (labels.current[3] = e)} position={[AX.cx + LOUPE.half, LOUPE.y - 0.45, 0]} tex="0.006" size={11} color="#8d929a" />
      <Label ref={(e) => void (labels.current[4] = e)} position={[AX.cx, AX.y + 0.82, 0]} tex={`\\pm\\sigma,\\ \\sigma = ${sd.toFixed(4)}`} size={14} color={C.norm} />
      <Label ref={(e) => void (labels.current[5] = e)} position={[AX.cx + 1.0 * S0 + 0.12, AX.y + 0.82, 0]} tex={`\\pm 1`} size={14} color={C.norm} align="left" />

      <Label ref={(e) => void (labels.current[6] = e)} position={[X_IN, capY, 0]} tex="x" size={18} />
      <Label ref={(e) => void (labels.current[7] = e)} position={[X_STAGE, capY, 0]} tex="x-\mu" size={18} color={C.norm} />
      <Label ref={(e) => void (labels.current[8] = e)} position={[X_STAGE, capY, 0]} tex="\hat x = \frac{x-\mu}{\sigma}" size={16} color={C.norm} />
      <Label ref={(e) => void (labels.current[9] = e)} position={[X_STAGE, capY, 0]} tex="\mathrm{LN}(x)" size={18} color={C.norm} />
      <Label ref={(e) => void (labels.current[10] = e)} position={[X_GAMMA, capY, 0]} tex="\gamma" size={18} color={C.norm} />
      <Label ref={(e) => void (labels.current[11] = e)} position={[X_BETA, capY, 0]} tex="\beta" size={18} color={C.norm} />
      <Label ref={(e) => void (labels.current[12] = e)} position={[X_IN, capY - 0.36, 0]} tex="768" size={12} color="#8d929a" />
      <Label ref={(e) => void (labels.current[13] = e)} position={[X_STAGE, capY - 0.36, 0]} tex="768" size={12} color="#8d929a" />
      <Label ref={(e) => void (labels.current[14] = e)} position={[X_GAMMA, capY - 0.36, 0]} tex="768" size={12} color="#8d929a" />
      <Label ref={(e) => void (labels.current[15] = e)} position={[X_BETA, capY - 0.36, 0]} tex="768" size={12} color="#8d929a" />

      <Label ref={(e) => void (labels.current[17] = e)} position={[AX.cx, -4.45, 0]} tex={`x - \\mu,\\quad \\mu = \\tfrac{1}{768}\\textstyle\\sum_i x_i = ${fmt4(mu)}`} size={18} />
      <Label ref={(e) => void (labels.current[18] = e)} position={[AX.cx, -4.45, 0]} tex={`\\hat x = \\frac{x-\\mu}{\\sigma},\\quad \\sigma = \\sqrt{\\tfrac{1}{768}\\textstyle\\sum_i (x_i-\\mu)^2+\\epsilon} = ${sd.toFixed(4)}`} size={18} />
      <Label ref={(e) => void (labels.current[19] = e)} position={[AX.cx, -4.45, 0]} tex={`\\mathrm{LN}(x) = \\gamma \\odot \\hat x + \\beta`} size={20} />
      <Label ref={(e) => void (labels.current[20] = e)} position={[0, columnBox(0, COL_TOP).y0 - 0.75, 0]} tex={`\\mathrm{LN}(X^{0}) \\in \\mathbb{R}^{${T}\\times 768}`} size={20} color={C.norm} />

      {[...AXIS_A, ...AXIS_B].map((v, j) => (
        <group key={j} ref={(g) => void (axisLabelGroups.current[j] = g)} position={[AX.cx, 0, 0]}>
          <Label ref={(e) => void (axisLabels.current[j] = e)} position={[0, AX.y - 0.66, 0]} tex={v < 0 ? `-${Math.abs(v)}` : `${v}`} size={12} color="#8d929a" />
        </group>
      ))}
    </group>
  );
}

/** Side view placement (world units). */
const GEO = { pos: [5.6, 1.7, 0] as [number, number, number], R: 1.0 };

/**
 * A small 3D picture of LayerNorm in d = 3: a point x, the diagonal (1,1,1), the plane through 0
 * perpendicular to it, and the sphere of radius √d. Centering moves x straight onto the plane;
 * rescaling moves it along the plane onto the sphere. (In GPT-2 the same happens in 768 dimensions.)
 */
function makeGeoInset() {
  // Display frame: the diagonal (1,1,1)/√3 is +Y, so the plane ⟂ to it is the XZ "floor".
  const root = new THREE.Group();
  const u0 = new THREE.Vector3(1, 1, 1).normalize();
  const e1 = new THREE.Vector3(1, -1, 0).normalize();
  const e2 = new THREE.Vector3().crossVectors(e1, u0).normalize();
  const toDisplay = (v: THREE.Vector3) => new THREE.Vector3(v.dot(e1), v.dot(u0), v.dot(e2));
  const R = GEO.R;
  const p0 = new THREE.Vector3(1.25, 0.95, 0.85);
  const pc = new THREE.Vector3(p0.x, 0, p0.z);
  const pn = pc.clone().setLength(R);
  const X = new THREE.Vector3(1, 0, 0), Y = new THREE.Vector3(0, 1, 0), Z = new THREE.Vector3(0, 0, 1);
  const lineObj = (pts: THREE.Vector3[], color: string, opacity: number) => {
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
    const l = new THREE.LineSegments(g, m);
    l.userData.base = opacity;
    l.renderOrder = 6;
    root.add(l);
    return l;
  };
  const circle = (a: THREE.Vector3, b: THREE.Vector3, r: number, n = 96) => {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < n; i++) {
      const t0 = (i / n) * Math.PI * 2, t1 = ((i + 1) / n) * Math.PI * 2;
      pts.push(a.clone().multiplyScalar(Math.cos(t0) * r).add(b.clone().multiplyScalar(Math.sin(t0) * r)));
      pts.push(a.clone().multiplyScalar(Math.cos(t1) * r).add(b.clone().multiplyScalar(Math.sin(t1) * r)));
    }
    return pts;
  };
  // the three coordinate axes of R³, seen in this frame
  const axes = [X, Y, Z].flatMap((a) => {
    const d = toDisplay(a);
    return [d.clone().multiplyScalar(-1.1), d.clone().multiplyScalar(1.1)];
  });
  lineObj(axes, C.dim, 0.3);
  // the diagonal direction (1,1,1)
  lineObj([Y.clone().multiplyScalar(-1.3), Y.clone().multiplyScalar(1.5)], C.dim, 0.7);
  // plane ⟂ (1,1,1): a disc outline and two chords
  const plane = lineObj([...circle(X, Z, 1.6), X.clone().multiplyScalar(-1.6), X.clone().multiplyScalar(1.6), Z.clone().multiplyScalar(-1.6), Z.clone().multiplyScalar(1.6)], C.norm, 0.3);
  // sphere of radius R: two faint great circles and the bright one where it meets the plane
  const sphere = lineObj([...circle(X, Y, R, 64), ...circle(Z, Y, R, 64)], C.norm, 0.16);
  const ring = lineObj(circle(X, Z, R), C.norm, 0.9);
  // paths: x → its projection (straight down the diagonal), projection → sphere (radially)
  const drop = lineObj([p0, pc], C.token, 0.55);
  const radial = lineObj([pc, pn], C.norm, 0.6);
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.065, 16, 12), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, depthWrite: false }));
  dot.renderOrder = 7;
  root.add(dot);
  let rotGroup: THREE.Group | null = null;
  return {
    root,
    diagTip: [0.12, 1.55, 0] as [number, number, number],
    sphereTag: [-R * 0.75, R * 0.75, 0] as [number, number, number],
    setRoot: (g: THREE.Group | null) => void (rotGroup = g),
    update(t1: number, t2: number, a: number, q: number) {
      if (rotGroup) {
        rotGroup.visible = a > 0.002;
        rotGroup.rotation.set(0.36, -0.75 + 0.6 * q, 0);
      }
      const pos = p0.clone().lerp(pc, t1).lerp(pn, t2);
      dot.position.copy(pos);
      (dot.material as THREE.MeshBasicMaterial).opacity = a;
      root.children.forEach((c) => {
        if (c === dot) return;
        const m = (c as THREE.LineSegments).material as THREE.LineBasicMaterial;
        let k = 1;
        if (c === drop) k = t1 > 0.001 ? 1 : 0;
        if (c === radial) k = t2 > 0.001 ? 1 : 0;
        if (c === ring || c === sphere) k = 0.35 + 0.65 * t2;
        if (c === plane) k = 0.5 + 0.5 * Math.max(t1, 1 - t2);
        m.opacity = (c.userData.base as number) * a * k;
      });
      // the drop line grows with the motion
      (drop.geometry as THREE.BufferGeometry).setFromPoints([p0, p0.clone().lerp(pc, t1)]);
      (radial.geometry as THREE.BufferGeometry).setFromPoints([pc, pc.clone().lerp(pn, t2)]);
    },
  };
}
