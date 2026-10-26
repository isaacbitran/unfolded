// Instanced glyph renderer: every glyph is one instance of a unit quad. Each instance has a
// target position and a "from" position; the vertex shader interpolates between them using a
// shared progress uniform and a per-instance delay window, so thousands of digits can fly from
// one tensor to another in a single draw call.

import { forwardRef, useEffect, useImperativeHandle, useMemo, useState } from 'react';
import * as THREE from 'three';
import { loadAtlas, type Atlas } from './atlas';

export interface GlyphItem {
  ch: string;
  x: number;
  y: number;
  z: number;
  /** glyph height in world units */
  h: number;
  color: THREE.Color;
  /** brightness multiplier (values > 1 bloom) */
  k: number;
  alpha: number;
  from?: [number, number, number];
  /** fraction of the flight progress at which this glyph starts / finishes moving */
  delay?: [number, number];
  /** groups glyphs so a scene can fade parts independently (0..3) */
  group?: number;
  /** fade in over [appear, appear+0.04] and out over [vanish−0.04, vanish] of the progress */
  appear?: number;
  vanish?: number;
}

export interface GlyphFieldHandle {
  mesh: THREE.Mesh | null;
  /** flight progress, 0 = at `from`, 1 = at target */
  setProgress(p: number): void;
  setOpacity(a: number): void;
  /** per-group opacity multipliers */
  setGroupOpacity(g: number, a: number): void;
  setArc(a: number): void;
}

const vert = /* glsl */ `
attribute vec3 aPos;
attribute vec3 aFrom;
attribute vec4 aUV;
attribute vec2 aSize;
attribute vec4 aColor;
attribute vec3 aDelay; // start, end, group
attribute vec2 aFade;  // appear, vanish
uniform float uProgress;
uniform float uArc;
uniform vec4 uGroup;
varying vec2 vUv;
varying vec4 vColor;
float ease(float t) { return t < 0.5 ? 4.0 * t * t * t : 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0; }
void main() {
  float t = clamp((uProgress - aDelay.x) / max(aDelay.y - aDelay.x, 1e-4), 0.0, 1.0);
  float e = ease(t);
  vec3 p = mix(aFrom, aPos, e);
  p.z += sin(e * 3.14159265) * uArc;
  vec3 local = p + vec3(position.xy * aSize, 0.0);
  vUv = aUV.xy + uv * aUV.zw;
  int g = int(aDelay.z + 0.5);
  float ga = g == 0 ? uGroup.x : g == 1 ? uGroup.y : g == 2 ? uGroup.z : uGroup.w;
  float fade = smoothstep(aFade.x, aFade.x + 0.04, uProgress) * (1.0 - smoothstep(aFade.y - 0.04, aFade.y, uProgress));
  vColor = vec4(aColor.rgb, aColor.a * ga * fade);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(local, 1.0);
}`;

const frag = /* glsl */ `
uniform sampler2D uAtlas;
uniform float uOpacity;
varying vec2 vUv;
varying vec4 vColor;
void main() {
  float a = texture2D(uAtlas, vUv, -0.6).a * vColor.a * uOpacity;
  if (a < 0.003) discard;
  gl_FragColor = vec4(vColor.rgb * a, a);
}`;

let atlasCache: Atlas | null = null;
export function useAtlas(): Atlas | null {
  const [atlas, setAtlas] = useState<Atlas | null>(atlasCache);
  useEffect(() => {
    if (!atlasCache) loadAtlas().then((a) => setAtlas((atlasCache = a)));
  }, []);
  return atlas;
}

function buildGeometry(items: GlyphItem[], atlas: Atlas) {
  const g = new THREE.InstancedBufferGeometry();
  const quad = new THREE.PlaneGeometry(1, 1);
  g.index = quad.index;
  g.setAttribute('position', quad.getAttribute('position'));
  g.setAttribute('uv', quad.getAttribute('uv'));
  const n = items.length;
  const aPos = new Float32Array(n * 3);
  const aFrom = new Float32Array(n * 3);
  const aUV = new Float32Array(n * 4);
  const aSize = new Float32Array(n * 2);
  const aColor = new Float32Array(n * 4);
  const aDelay = new Float32Array(n * 3);
  const aFade = new Float32Array(n * 2);
  const fallback = atlas.glyphs.get('?')!;
  items.forEach((it, i) => {
    const gl = atlas.glyphs.get(it.ch) ?? fallback;
    aPos.set([it.x, it.y, it.z], i * 3);
    aFrom.set(it.from ?? [it.x, it.y, it.z], i * 3);
    aUV.set(gl.uv, i * 4);
    aSize.set([it.h * atlas.aspect, it.h], i * 2);
    aColor.set([it.color.r * it.k, it.color.g * it.k, it.color.b * it.k, it.alpha], i * 4);
    aDelay.set([it.delay?.[0] ?? 0, it.delay?.[1] ?? 1, it.group ?? 0], i * 3);
    aFade.set([it.appear ?? -1, it.vanish ?? 3], i * 2);
  });
  g.setAttribute('aPos', new THREE.InstancedBufferAttribute(aPos, 3));
  g.setAttribute('aFrom', new THREE.InstancedBufferAttribute(aFrom, 3));
  g.setAttribute('aUV', new THREE.InstancedBufferAttribute(aUV, 4));
  g.setAttribute('aSize', new THREE.InstancedBufferAttribute(aSize, 2));
  g.setAttribute('aColor', new THREE.InstancedBufferAttribute(aColor, 4));
  g.setAttribute('aDelay', new THREE.InstancedBufferAttribute(aDelay, 3));
  g.setAttribute('aFade', new THREE.InstancedBufferAttribute(aFade, 2));
  g.instanceCount = n;
  // generous bounds: flights travel between distant objects
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
  return g;
}

export const GlyphField = forwardRef<GlyphFieldHandle, { items: GlyphItem[]; renderOrder?: number }>(function GlyphField(
  { items, renderOrder = 10 },
  ref,
) {
  const atlas = useAtlas();
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: vert,
        fragmentShader: frag,
        transparent: true,
        depthWrite: false,
        depthTest: true,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneMinusSrcAlphaFactor,
        uniforms: {
          uAtlas: { value: null },
          uProgress: { value: 1 },
          uOpacity: { value: 1 },
          uArc: { value: 0 },
          uGroup: { value: new THREE.Vector4(1, 1, 1, 1) },
        },
      }),
    [],
  );
  useEffect(() => {
    if (atlas) material.uniforms.uAtlas.value = atlas.texture;
  }, [atlas, material]);
  const geometry = useMemo(() => (atlas ? buildGeometry(items, atlas) : null), [items, atlas]);
  useEffect(() => () => geometry?.dispose(), [geometry]);
  const [mesh, setMesh] = useState<THREE.Mesh | null>(null);
  useImperativeHandle(
    ref,
    () => ({
      mesh,
      setProgress: (p) => (material.uniforms.uProgress.value = p),
      setOpacity: (a) => {
        material.uniforms.uOpacity.value = a;
        if (mesh) mesh.visible = a > 0.002;
      },
      setGroupOpacity: (g, a) => ((material.uniforms.uGroup.value as THREE.Vector4).setComponent(g, a)),
      setArc: (a) => (material.uniforms.uArc.value = a),
    }),
    [material, mesh],
  );
  if (!geometry) return null;
  return <mesh ref={setMesh} geometry={geometry} material={material} frustumCulled={false} renderOrder={renderOrder} />;
});

// ---------- text layout ----------

let measureAtlas: Atlas | null = null;
loadAtlas().then((a) => (measureAtlas = a));

/** Width of a string in world units at glyph height h. */
export function textWidth(s: string, h: number): number {
  if (!measureAtlas) return s.length * h * 0.5;
  let w = 0;
  for (const ch of s) w += (measureAtlas.glyphs.get(ch)?.advance ?? 0.5) * h;
  return w;
}

export interface TextOpts {
  h: number;
  color: THREE.Color;
  k?: number;
  alpha?: number;
  align?: 'left' | 'center' | 'right';
  group?: number;
  from?: [number, number, number];
  delay?: [number, number];
  appear?: number;
  vanish?: number;
}

/** Lay out a string; returns one GlyphItem per non-space character. */
export function layoutText(s: string, x: number, y: number, z: number, o: TextOpts): GlyphItem[] {
  const out: GlyphItem[] = [];
  const width = textWidth(s, o.h);
  let cx = o.align === 'left' ? x : o.align === 'right' ? x - width : x - width / 2;
  for (const ch of s) {
    const adv = (measureAtlas?.glyphs.get(ch)?.advance ?? 0.5) * o.h;
    if (ch !== ' ') {
      const gx = cx + adv / 2;
      out.push({
        ch, x: gx, y, z, h: o.h, color: o.color, k: o.k ?? 1, alpha: o.alpha ?? 1, group: o.group, delay: o.delay, appear: o.appear, vanish: o.vanish,
        from: o.from ? [o.from[0] + gx - x, o.from[1], o.from[2]] : undefined,
      });
    }
    cx += adv;
  }
  return out;
}
