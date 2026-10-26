// The whole model in depth: one residual rail per token running along −z through 12 blocks, each
// block an Attention slab (purple) and an MLP slab (gold), with every block's real attention
// patterns as small heat grids and its MLP firing rate. Shared by the intro, residual-stream,
// stack and autoregression steps and by explore mode.

import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { C, colX } from '../theme';
import { Glass, type GlassHandle } from './Glass';
import { Label, fade } from './Label';

export const TOWER = {
  blockGap: 14,
  /** z of block L's center (0-based) */
  blockZ: (L: number) => -14 * (L + 1),
  attnZ: (L: number) => -14 * (L + 1) + 3.2,
  mlpZ: (L: number) => -14 * (L + 1) - 3.2,
  headZ: -182,
  railY: 0,
  slabH: 3.2,
  slabD: 2.6,
};

/** ‖x‖ of each token's residual stream after each of the 13 stages. [stage][token] */
export function streamNorms(trace: SceneTrace): number[][] {
  return trace.resid.map((r) => trace.tokens.map((_, t) => {
    let s = 0;
    for (let i = 0; i < 768; i++) s += r[t * 768 + i] ** 2;
    return Math.sqrt(s);
  }));
}

/** Fraction of MLP neurons with GELU output > 0 per layer, averaged over tokens. */
export function firingRates(trace: SceneTrace): number[] {
  return trace.layers.map((l) => {
    let on = 0;
    for (let i = 0; i < l.mlpPost.length; i++) if (l.mlpPost[i] > 0) on++;
    return on / l.mlpPost.length;
  });
}

const railVert = /* glsl */ `
attribute float aZ;
attribute float aGlow;
varying float vZ;
varying float vGlow;
void main() {
  vZ = aZ; vGlow = aGlow;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const railFrag = /* glsl */ `
uniform float uOpacity;
uniform float uReach;   // rails are drawn from z = 2 down to z = uReach
uniform float uPulse;   // z of the travelling pulse
uniform float uPulseK;
uniform vec3 uColor;
varying float vZ;
varying float vGlow;
void main() {
  if (vZ < uReach) discard;
  float pulse = uPulseK * exp(-pow((vZ - uPulse) / 2.2, 2.0));
  float a = uOpacity * (0.18 + 0.55 * vGlow + pulse);
  gl_FragColor = vec4(uColor * a * (1.0 + pulse * 2.0), a);
}`;

export interface TowerHandle {
  setOpacity(a: number): void;
  /** rails reach from the front down to z (0 → nothing, TOWER.headZ → everything) */
  setReach(z: number): void;
  setPulse(z: number, k: number): void;
  setBlocksOpacity(a: number): void;
  setPatternsOpacity(a: number): void;
  setLabelsOpacity(a: number): void;
  group: THREE.Group | null;
}

export interface TowerProps {
  trace: SceneTrace;
  /** number of token slots to lay out (≥ T; lets autoregression add a column without moving the others) */
  slots?: number;
}

export const Tower = forwardRef<TowerHandle, TowerProps>(function Tower({ trace, slots }, ref) {
  const T = trace.T;
  const S = Math.max(slots ?? T, T);
  const group = useRef<THREE.Group>(null);
  const attn = useRef<(GlassHandle | null)[]>([]);
  const mlp = useRef<(GlassHandle | null)[]>([]);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const width = S * 2.0 + 1.2;

  const norms = useMemo(() => streamNorms(trace), [trace]);
  const maxNorm = useMemo(() => Math.max(...norms.flat()), [norms]);

  // Rails: one segment per (token, stage) with glow ∝ log of the stream's norm at that depth.
  const rail = useMemo(() => {
    const pos: number[] = [];
    const z: number[] = [];
    const glow: number[] = [];
    for (let t = 0; t < T; t++) {
      const x = colX(t, S);
      const stops = [2, ...Array.from({ length: 12 }, (_, L) => TOWER.mlpZ(L) - 1.5), TOWER.headZ + 4];
      for (let s = 0; s < stops.length - 1; s++) {
        const g = Math.log1p(norms[Math.min(s, 12)][t]) / Math.log1p(maxNorm);
        pos.push(x, TOWER.railY, stops[s], x, TOWER.railY, stops[s + 1]);
        z.push(stops[s], stops[s + 1]);
        glow.push(g, g);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('aZ', new THREE.Float32BufferAttribute(z, 1));
    geo.setAttribute('aGlow', new THREE.Float32BufferAttribute(glow, 1));
    return { geo };
  }, [trace, T, S, norms, maxNorm]);
  const railMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: railVert,
        fragmentShader: railFrag,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uOpacity: { value: 1 },
          uReach: { value: TOWER.headZ },
          uPulse: { value: 1e3 },
          uPulseK: { value: 0 },
          uColor: { value: new THREE.Color(C.token) },
        },
      }),
    [],
  );

  // Attention patterns: 12 heads × T×T cells per block, laid out as a 6×2 grid on the slab's front face.
  const patterns = useMemo(() => {
    const cell = Math.min(0.11, (width - 1.6) / 6 / (T + 1.5));
    const n = 12 * 12 * T * T;
    const geo = new THREE.PlaneGeometry(cell * 0.92, cell * 0.92);
    const mat = new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 1, toneMapped: false });
    const mesh = new THREE.InstancedMesh(geo, mat, n);
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    const purple = new THREE.Color(C.attn);
    let i = 0;
    const gridW = T * cell;
    const gapX = (width - 1.6 - 6 * gridW) / 5;
    for (let L = 0; L < 12; L++) {
      const z = TOWER.attnZ(L) + TOWER.slabD / 2 + 0.02;
      for (let h = 0; h < 12; h++) {
        const gx = -width / 2 + 0.8 + (h % 6) * (gridW + gapX);
        const gy = TOWER.railY + 1.3 - Math.floor(h / 6) * (gridW + 0.25);
        const A = trace.layers[L].pattern;
        for (let r = 0; r < T; r++)
          for (let k = 0; k < T; k++) {
            const a = A[(h * T + r) * T + k];
            m.makeTranslation(gx + (k + 0.5) * cell, gy - (r + 0.5) * cell, z);
            mesh.setMatrixAt(i, m);
            c.copy(purple).multiplyScalar(k > r ? 0.02 : 0.08 + 1.5 * Math.sqrt(a));
            mesh.setColorAt(i, c);
            i++;
          }
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.frustumCulled = false;
    mesh.renderOrder = 6;
    return mesh;
  }, [trace, T, width]);

  const rates = useMemo(() => firingRates(trace), [trace]);

  useImperativeHandle(ref, () => ({
    group: group.current,
    setOpacity(a) {
      railMat.uniforms.uOpacity.value = a;
      if (group.current) group.current.visible = a > 0.002;
    },
    setReach: (z) => (railMat.uniforms.uReach.value = z),
    setPulse(z, k) {
      railMat.uniforms.uPulse.value = z;
      railMat.uniforms.uPulseK.value = k;
    },
    setBlocksOpacity(a) {
      attn.current.forEach((g) => g?.setOpacity(a));
      mlp.current.forEach((g) => g?.setOpacity(a));
    },
    setPatternsOpacity(a) {
      (patterns.material as THREE.MeshBasicMaterial).opacity = a;
      patterns.visible = a > 0.002;
    },
    setLabelsOpacity: (a) => labels.current.forEach((l) => fade(l, a)),
  }));

  return (
    <group ref={group}>
      <lineSegments geometry={rail.geo} material={railMat} renderOrder={3} frustumCulled={false} />
      {Array.from({ length: 12 }, (_, L) => (
        <group key={L}>
          <Glass ref={(g) => void (attn.current[L] = g)} size={[width, TOWER.slabH, TOWER.slabD]} position={[colX(0, S) + ((S - 1) * 2.0) / 2, TOWER.railY + 0.6, TOWER.attnZ(L)]} color={C.attn} fill={0.006} rim={0.22} edge={0.4} />
          <Glass ref={(g) => void (mlp.current[L] = g)} size={[width, TOWER.slabH * 0.8, TOWER.slabD]} position={[colX(0, S) + ((S - 1) * 2.0) / 2, TOWER.railY + 0.4, TOWER.mlpZ(L)]} color={C.mlp} fill={0.004 + 0.012 * rates[L]} rim={0.2} edge={0.35} />
          <Label ref={(e) => void (labels.current[L] = e)} position={[colX(0, S) - 1.6, TOWER.railY + 2.4, TOWER.blockZ(L)]} text={`block ${L + 1}`} size={12} color="#8d929a" align="right" />
        </group>
      ))}
      <primitive object={patterns} />
    </group>
  );
});
