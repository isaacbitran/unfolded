// Step 11 — the MLP, for the last token only. LN₂'s output a (silver) flows into W_up (a wide gold
// slab, 768 × 3072), and 3,072 neurons drop out of it into a 48 × 64 field coloured by their real
// pre-activations. GELU then dims or lights each neuron while a copy of every neuron lands on the
// GELU curve at (x, GELU(x)). Finally the field collapses into W_down (3072 × 768, tall), the
// 768-number output appears, and it is added to the residual stream: x_mid + MLP(a) = x¹.
//
// Also exports `Quads`, the instanced dot/bar helper used by steps 11, 13, 14 and 16.

import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { gelu } from '../../model/gpt2';
import type { SceneTrace } from '../../model/trace';
import { Columns, type ColumnsHandle } from '../primitives/Columns';
import { Lines } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { MatrixSlab, type MatrixSlabHandle } from '../primitives/MatrixSlab';
import { TextPlane, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { C, L, easeInOut, seg } from '../theme';
import { useStep } from '../useStep';
import { columnBox } from '../vectors';

// ---------------------------------------------------------------------------------------------
// Quads: one instanced draw call for thousands of dots or bars. Each instance has a target
// rectangle and a "from" rectangle (position and size), a delay window for the flight, and two
// colours A → B blended by a sweep uniform (instance switches when uMix passes its key).

export interface QuadItem {
  x: number;
  y: number;
  z?: number;
  w: number;
  h: number;
  /** flight origin (centre); defaults to the target */
  from?: [number, number, number];
  /** size at the origin; defaults to the target size */
  fromSize?: [number, number];
  /** colour A (linear rgb already multiplied by brightness) and alpha */
  a: [number, number, number, number];
  /** colour B; defaults to A */
  b?: [number, number, number, number];
  delay?: [number, number];
  /** sweep key: colour is B once uMix > key */
  key?: number;
}

export interface QuadsHandle {
  mesh: THREE.Mesh | null;
  setProgress(p: number): void;
  setOpacity(a: number): void;
  setMix(m: number): void;
  setArc(a: number): void;
  /** overwrite instance i's target and origin rectangle (static, no flight) */
  setRect(i: number, x: number, y: number, w: number, h: number): void;
  setColor(i: number, r: number, g: number, b: number, a: number): void;
  /** push CPU-side edits to the GPU */
  commit(): void;
}

const quadVert = /* glsl */ `
attribute vec3 aPos;
attribute vec3 aFrom;
attribute vec4 aSize;   // target w, h, from w, h
attribute vec4 aColA;
attribute vec4 aColB;
attribute vec3 aDelay;  // start, end, sweep key
uniform float uProgress;
uniform float uArc;
uniform float uMix;
varying vec2 vUv;
varying vec4 vColor;
float ease(float t) { return t < 0.5 ? 4.0 * t * t * t : 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0; }
void main() {
  float t = clamp((uProgress - aDelay.x) / max(aDelay.y - aDelay.x, 1e-4), 0.0, 1.0);
  float e = ease(t);
  vec3 p = mix(aFrom, aPos, e);
  p.z += sin(e * 3.14159265) * uArc;
  vec2 s = mix(aSize.zw, aSize.xy, e);
  vUv = uv;
  float m = smoothstep(aDelay.z, aDelay.z + 0.08, uMix);
  vColor = mix(aColA, aColB, m);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p + vec3(position.xy * s, 0.0), 1.0);
}`;

const quadFrag = /* glsl */ `
uniform float uOpacity;
uniform float uRound;
varying vec2 vUv;
varying vec4 vColor;
void main() {
  float a = vColor.a * uOpacity;
  if (uRound > 0.5) a *= smoothstep(0.5, 0.36, length(vUv - 0.5));
  if (a < 0.003) discard;
  gl_FragColor = vec4(vColor.rgb * a, a);
}`;

export const Quads = forwardRef<QuadsHandle, { items: QuadItem[]; round?: boolean; renderOrder?: number }>(function Quads(
  { items, round = false, renderOrder = 8 },
  ref,
) {
  const mesh = useRef<THREE.Mesh>(null);
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: quadVert,
        fragmentShader: quadFrag,
        transparent: true,
        depthWrite: false,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneMinusSrcAlphaFactor,
        uniforms: {
          uProgress: { value: 1 },
          uOpacity: { value: 0 },
          uArc: { value: 0 },
          uMix: { value: 0 },
          uRound: { value: round ? 1 : 0 },
        },
      }),
    [round],
  );
  const geometry = useMemo(() => {
    const g = new THREE.InstancedBufferGeometry();
    const quad = new THREE.PlaneGeometry(1, 1);
    g.index = quad.index;
    g.setAttribute('position', quad.getAttribute('position'));
    g.setAttribute('uv', quad.getAttribute('uv'));
    const n = items.length;
    const aPos = new Float32Array(n * 3);
    const aFrom = new Float32Array(n * 3);
    const aSize = new Float32Array(n * 4);
    const aColA = new Float32Array(n * 4);
    const aColB = new Float32Array(n * 4);
    const aDelay = new Float32Array(n * 3);
    items.forEach((it, i) => {
      const z = it.z ?? 0;
      aPos.set([it.x, it.y, z], i * 3);
      aFrom.set(it.from ?? [it.x, it.y, z], i * 3);
      aSize.set([it.w, it.h, it.fromSize?.[0] ?? it.w, it.fromSize?.[1] ?? it.h], i * 4);
      aColA.set(it.a, i * 4);
      aColB.set(it.b ?? it.a, i * 4);
      aDelay.set([it.delay?.[0] ?? 0, it.delay?.[1] ?? 1e-3, it.key ?? 2], i * 3);
    });
    g.setAttribute('aPos', new THREE.InstancedBufferAttribute(aPos, 3));
    g.setAttribute('aFrom', new THREE.InstancedBufferAttribute(aFrom, 3));
    g.setAttribute('aSize', new THREE.InstancedBufferAttribute(aSize, 4));
    g.setAttribute('aColA', new THREE.InstancedBufferAttribute(aColA, 4));
    g.setAttribute('aColB', new THREE.InstancedBufferAttribute(aColB, 4));
    g.setAttribute('aDelay', new THREE.InstancedBufferAttribute(aDelay, 3));
    g.instanceCount = n;
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
    return g;
  }, [items]);
  useImperativeHandle(
    ref,
    () => ({
      mesh: mesh.current,
      setProgress: (p) => void (material.uniforms.uProgress.value = p),
      setOpacity: (a) => {
        material.uniforms.uOpacity.value = a;
        if (mesh.current) mesh.current.visible = a > 0.002;
      },
      setMix: (m) => void (material.uniforms.uMix.value = m),
      setArc: (a) => void (material.uniforms.uArc.value = a),
      setRect(i, x, y, w, h) {
        const pos = geometry.getAttribute('aPos') as THREE.InstancedBufferAttribute;
        const from = geometry.getAttribute('aFrom') as THREE.InstancedBufferAttribute;
        const size = geometry.getAttribute('aSize') as THREE.InstancedBufferAttribute;
        pos.setXY(i, x, y);
        from.setXY(i, x, y);
        size.setXYZW(i, w, h, w, h);
      },
      setColor(i, r, g, b, a) {
        (geometry.getAttribute('aColA') as THREE.InstancedBufferAttribute).setXYZW(i, r, g, b, a);
        (geometry.getAttribute('aColB') as THREE.InstancedBufferAttribute).setXYZW(i, r, g, b, a);
      },
      commit() {
        for (const k of ['aPos', 'aFrom', 'aSize', 'aColA', 'aColB']) geometry.getAttribute(k).needsUpdate = true;
      },
    }),
    [material, geometry],
  );
  return <mesh ref={mesh} geometry={geometry} material={material} frustumCulled={false} renderOrder={renderOrder} visible={false} />;
});

/** Linear rgb of a theme colour times a brightness, plus alpha. */
export function rgba(hex: string, k: number, a = 1): [number, number, number, number] {
  const c = new THREE.Color(hex);
  return [c.r * k, c.g * k, c.b * k, a];
}

/** A polyline drawn on progressively (setDraw 0..1). */
export const Curve = forwardRef<{ setDraw(t: number): void; setOpacity(a: number): void }, { points: number[]; color: string; opacity?: number }>(
  function Curve({ points, color, opacity = 0.9 }, ref) {
    const line = useRef<THREE.Line>(null);
    const geo = useMemo(() => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
      return g;
    }, [points]);
    const mat = useMemo(() => new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false }), [color]);
    const n = points.length / 3;
    useImperativeHandle(ref, () => ({
      setDraw: (t) => geo.setDrawRange(0, Math.max(0, Math.round(n * t))),
      setOpacity: (a) => {
        mat.opacity = opacity * a;
        if (line.current) line.current.visible = a > 0.002;
      },
    }), [geo, mat, n, opacity]);
    return <primitive object={useMemo(() => new THREE.Line(geo, mat), [geo, mat])} ref={line} renderOrder={6} />;
  },
);

// ---------------------------------------------------------------------------------------------

const UP = { x: 0, y: 3.3, w: 8.0, h: 2.0 };
const FIELD = { cols: 64, rows: 48, s: 0.12, cx: 0, cy: -1.4 };
const DOWN = { x: 6.3, y: -1.4, w: 1.6, h: 6.4 };
const A_X = -6.4;
const A_TOP = UP.y + (L.visible + 1) * L.rowH * 0.5;
const OUT_X = 8.5;
const MID_X = 10.4;
const OUT_TOP = 0.2;
// GELU plot: x-domain [−5.5, 5], y-domain [−0.5, 5], same scale on both axes
const PLOT = { ox: 7.4, oy: -2.6, k: 0.5, xmin: -5.5, xmax: 5, ymax: 5 };
const plotX = (v: number) => PLOT.ox + v * PLOT.k;
const plotY = (v: number) => PLOT.oy + v * PLOT.k;

const fieldX = (c: number) => FIELD.cx + (c - (FIELD.cols - 1) / 2) * FIELD.s;
const fieldY = (r: number) => FIELD.cy + ((FIELD.rows - 1) / 2 - r) * FIELD.s;

export function S11MLP({ index, trace }: { index: number; trace: SceneTrace }) {
  const T = trace.T;
  const t = T - 1;
  const layer = trace.layers[0];
  const pre = useMemo(() => layer.mlpPre.subarray(t * 3072, (t + 1) * 3072), [layer, t]);
  const nPos = useMemo(() => pre.reduce((n, v) => n + (v > 0 ? 1 : 0), 0), [pre]);

  const up = useRef<MatrixSlabHandle>(null);
  const down = useRef<MatrixSlabHandle>(null);
  const aCol = useRef<ColumnsHandle>(null);
  const aFly = useRef<ColumnsHandle>(null);
  const field = useRef<QuadsHandle>(null);
  const scatter = useRef<QuadsHandle>(null);
  const sink = useRef<QuadsHandle>(null);
  const outCol = useRef<ColumnsHandle>(null);
  const outGroup = useRef<THREE.Group>(null);
  const midCol = useRef<ColumnsHandle>(null);
  const sumCol = useRef<ColumnsHandle>(null);
  const curve = useRef<{ setDraw(t: number): void; setOpacity(a: number): void }>(null);
  const axes = useRef<THREE.LineSegments>(null);
  const tok = useRef<TextPlaneHandle>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);

  const aXs = useMemo(() => [A_X], []);
  // the flying copy of a ends on the slab's left edge, one row per entry
  const aFlyFrom = useMemo(() => [[A_X, A_TOP, 0] as [number, number, number]], []);
  const aDelay = useMemo(() => (_c: number, r: number) => [0.1 + r * 0.008, 0.2 + r * 0.008] as [number, number], []);
  const outFrom = useMemo(() => [[DOWN.x + DOWN.w / 2 + 0.7, OUT_TOP + 0.4, 0] as [number, number, number]], []);
  const outDelay = useMemo(() => (_c: number, r: number) => [0.73 + r * 0.006, 0.81 + r * 0.006] as [number, number], []);

  const { fieldItems, scatterItems, sinkItems } = useMemo(() => {
    const fieldItems: QuadItem[] = [];
    const scatterItems: QuadItem[] = [];
    const sinkItems: QuadItem[] = [];
    const slabBottom = UP.y - UP.h / 2;
    for (let j = 0; j < 3072; j++) {
      const r = Math.floor(j / FIELD.cols);
      const c = j % FIELD.cols;
      const x = fieldX(c);
      const y = fieldY(r);
      const v = pre[j];
      const g = gelu(v);
      const colA = v > 0 ? rgba(C.mlp, 0.55 + 1.1 * Math.min(1, v / 2.5)) : rgba(C.dim, 0.12 + 0.3 * Math.min(1, -v / 3));
      const colB = g > 0.02 ? rgba(C.mlp, 0.5 + 1.4 * Math.min(1, g / 2.5)) : rgba(C.dim, 0.1);
      const d0 = 0.2 + (r / FIELD.rows) * 0.1 + (c / FIELD.cols) * 0.02;
      fieldItems.push({ x, y, w: 0.085, h: 0.085, from: [x, slabBottom + 0.05, 0], fromSize: [0.02, 0.02], a: colA, b: colB, delay: [d0, d0 + 0.06], key: 0.05 + (c / FIELD.cols) * 0.8 });
      const s0 = 0.4 + (c / FIELD.cols) * 0.08;
      scatterItems.push({ x: plotX(Math.max(PLOT.xmin, v)), y: plotY(g), w: 0.06, h: 0.06, from: [x, y, 0], fromSize: [0.085, 0.085], a: v > 0 ? rgba(C.mlp, 1.3) : rgba(C.mlp, 0.6), delay: [s0, s0 + 0.08] });
      const k0 = 0.62 + (c / FIELD.cols) * 0.06 + (r / FIELD.rows) * 0.02;
      sinkItems.push({ x: DOWN.x - DOWN.w / 2 + 0.05, y: DOWN.y + DOWN.h / 2 - 0.2 - (j / 3071) * (DOWN.h - 0.4), w: 0.03, h: 0.03, from: [x, y, 0], fromSize: [0.085, 0.085], a: colB, delay: [k0, k0 + 0.06] });
    }
    return { fieldItems, scatterItems, sinkItems };
  }, [pre]);

  const curvePts = useMemo(() => {
    const pts: number[] = [];
    for (let i = 0; i <= 200; i++) {
      const v = PLOT.xmin + ((PLOT.xmax - PLOT.xmin) * i) / 200;
      pts.push(plotX(v), plotY(gelu(v)), 0.01);
    }
    return pts;
  }, []);
  const axisPts = useMemo(
    () => [plotX(PLOT.xmin), plotY(0), 0, plotX(PLOT.xmax), plotY(0), 0, plotX(0), plotY(-0.4), 0, plotX(0), plotY(PLOT.ymax), 0],
    [],
  );

  useStep(index, (sp, vis) => {
    // the action completes by 88% of the step; the camera glides to the next step after 86%
    const p = Math.min(1, sp / 0.88);
    const on = vis > 0.001;
    // phase A: a enters W_up, neurons drop into the field
    const upA = vis * seg(p, 0.02, 0.1) * (1 - seg(p, 0.56, 0.64));
    up.current?.setOpacity(upA);
    const sweep = seg(p, 0.1, 0.24);
    up.current?.setBand(UP.h / 2 - sweep * UP.h, 0.18, on ? 0.35 * Math.sin(Math.PI * sweep) : 0);
    tok.current?.setOpacity(vis * seg(p, 0, 0.08) * (1 - seg(p, 0.22, 0.27)));
    aCol.current?.setOpacity(vis * seg(p, 0, 0.08) * (1 - 0.85 * seg(p, 0.1, 0.16)) * (1 - seg(p, 0.22, 0.27)));
    aFly.current?.setProgress(p);
    aFly.current?.setOpacity(on ? vis * (seg(p, 0.1, 0.11) - seg(p, 0.19, 0.25)) : 0);
    aFly.current?.setBracketOpacity(0);

    field.current?.setProgress(p);
    field.current?.setMix(seg(p, 0.4, 0.52));
    field.current?.setOpacity(vis * seg(p, 0.19, 0.21) * (1 - seg(p, 0.63, 0.7)));
    // phase B: GELU curve, scatter, fraction > 0
    const plotA = vis * seg(p, 0.36, 0.42) * (1 - seg(p, 0.55, 0.59));
    curve.current?.setOpacity(plotA);
    curve.current?.setDraw(easeInOut(seg(p, 0.36, 0.46)));
    if (axes.current) {
      (axes.current.material as THREE.LineBasicMaterial).opacity = 0.35 * plotA;
      axes.current.visible = plotA > 0.002;
    }
    scatter.current?.setProgress(p);
    scatter.current?.setArc(0.6);
    scatter.current?.setOpacity(on ? vis * seg(p, 0.4, 0.41) * (1 - seg(p, 0.55, 0.59)) * 0.9 : 0);
    // phase C: field collapses into W_down, output appears, residual add
    down.current?.setOpacity(vis * seg(p, 0.58, 0.63) * (1 - 0.6 * seg(p, 0.88, 0.96)));
    down.current?.setBand(DOWN.h / 2 - seg(p, 0.64, 0.74) * DOWN.h, 0.3, on ? 0.3 * Math.sin(Math.PI * seg(p, 0.64, 0.74)) : 0);
    sink.current?.setProgress(p);
    sink.current?.setArc(0.4);
    sink.current?.setOpacity(on ? vis * (seg(p, 0.62, 0.63) - seg(p, 0.72, 0.76)) : 0);
    outCol.current?.setProgress(p);
    outCol.current?.setArc(0.3);
    const merge = easeInOut(seg(p, 0.84, 0.92));
    const outFade = 1 - seg(p, 0.9, 0.95);
    outCol.current?.setOpacity(vis * seg(p, 0.72, 0.74) * outFade);
    outCol.current?.setBracketOpacity(vis * seg(p, 0.76, 0.8) * (1 - seg(p, 0.84, 0.88)));
    if (outGroup.current) outGroup.current.position.x = (MID_X - OUT_X) * merge;
    midCol.current?.setOpacity(vis * seg(p, 0.76, 0.82) * outFade);
    sumCol.current?.setOpacity(vis * seg(p, 0.9, 0.96));

    const lab = (i: number, a: number) => fade(labels.current[i], vis * a);
    lab(0, seg(p, 0.03, 0.1) * (1 - seg(p, 0.2, 0.25)));
    lab(1, seg(p, 0.24, 0.3) * (1 - seg(p, 0.38, 0.42)));
    lab(2, seg(p, 0.46, 0.52) * (1 - seg(p, 0.6, 0.64)));
    lab(3, seg(p, 0.4, 0.46) * (1 - seg(p, 0.55, 0.59)));
    lab(4, seg(p, 0.76, 0.82) * (1 - seg(p, 0.84, 0.88)));
    lab(5, seg(p, 0.92, 0.98));
    lab(6, seg(p, 0.79, 0.84) * (1 - seg(p, 0.84, 0.88)));
    lab(7, seg(p, 0.79, 0.84) * (1 - seg(p, 0.84, 0.88)));
  });

  const aBox = columnBox(A_X, A_TOP);
  const outBox = columnBox(OUT_X, OUT_TOP);
  const pct = ((100 * nPos) / 3072).toFixed(1);
  return (
    <group>
      <MatrixSlab ref={up} preview={trace.previews.W_up} position={[UP.x, UP.y, 0]} w={UP.w} h={UP.h} color={C.mlp} name="W_{\uparrow}" glyphH={0.17} />
      <MatrixSlab ref={down} preview={trace.previews.W_down} position={[DOWN.x, DOWN.y, 0]} w={DOWN.w} h={DOWN.h} color={C.mlp} name="W_{\downarrow}" numbers={false} />
      <TextPlane ref={tok} text={visibleToken(trace.tokens[t].text)} height={0.34} position={[A_X, A_TOP + 0.6, 0]} />
      <Columns ref={aCol} data={layer.ln2} D={768} xs={aXs} rows={[t]} y0={A_TOP} color={C.norm} />
      <Columns ref={aFly} data={layer.ln2} D={768} xs={[UP.x - UP.w / 2 - 0.45]} rows={[t]} y0={A_TOP} color={C.norm} from={aFlyFrom} delay={aDelay} />
      <Quads ref={field} items={fieldItems} round />
      <Lines ref={axes} points={axisPts} color={C.dim} opacity={0} />
      <Curve ref={curve} points={curvePts} color={C.mlp} />
      <Quads ref={scatter} items={scatterItems} round renderOrder={9} />
      <Quads ref={sink} items={sinkItems} round />
      <group ref={outGroup}>
        <Columns ref={outCol} data={layer.mlpOut} D={768} xs={[OUT_X]} rows={[t]} y0={OUT_TOP} color={C.mlp} from={outFrom} delay={outDelay} />
      </group>
      <Columns ref={midCol} data={layer.residMid} D={768} xs={[MID_X]} rows={[t]} y0={OUT_TOP} color={C.token} />
      <Columns ref={sumCol} data={trace.resid[1]} D={768} xs={[MID_X]} rows={[t]} y0={OUT_TOP} color={C.token} />

      <Label ref={(e) => void (labels.current[0] = e)} position={[A_X, aBox.y0 - 0.45, 0]} tex={`a=\\text{LN}_2(x_T)\\in\\mathbb{R}^{768}`} size={15} color="#cfd5dd" />
      <Label ref={(e) => void (labels.current[1] = e)} position={[FIELD.cx, fieldY(FIELD.rows - 1) - 0.55, 0]} tex={`h = a\\,W_{\\uparrow}+b_{\\uparrow}\\in\\mathbb{R}^{3072}\\quad(48\\times64\\text{ neurons})`} size={16} />
      <Label ref={(e) => void (labels.current[2] = e)} position={[FIELD.cx, fieldY(FIELD.rows - 1) - 0.55, 0]} tex={`${nPos}\\text{ of }3{,}072\\text{ neurons} > 0\\ \\ (${pct}\\%)`} size={16} />
      <Label ref={(e) => void (labels.current[3] = e)} position={[plotX(-2.6), plotY(3.6), 0]} tex={`\\text{GELU}(h)`} size={17} color={C.mlp} />
      <Label ref={(e) => void (labels.current[4] = e)} position={[OUT_X, outBox.y0 - 0.45, 0]} tex={`\\text{GELU}(h)\\,W_{\\downarrow}+b_{\\downarrow}`} size={14} color={C.mlp} />
      <Label ref={(e) => void (labels.current[5] = e)} position={[MID_X, outBox.y0 - 0.45, 0]} tex={`x^{1}_T = x^{\\text{mid}}_T + \\text{MLP}(a)\\in\\mathbb{R}^{768}`} size={15} />
      <Label ref={(e) => void (labels.current[6] = e)} position={[MID_X, outBox.y0 - 0.45, 0]} tex={`x^{\\text{mid}}_T`} size={15} />
      <Label ref={(e) => void (labels.current[7] = e)} position={[(OUT_X + MID_X) / 2, OUT_TOP - 1.2, 0]} tex="+" size={22} color="#d8d4ca" />
    </group>
  );
}
