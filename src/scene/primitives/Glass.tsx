// Glass slab: faint tinted fill, Fresnel rim, and crisp 1 px edges. Cheap enough for dozens of slabs.

import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import * as THREE from 'three';

const vert = /* glsl */ `
varying vec3 vN;
varying vec3 vView;
varying vec3 vLocal;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vView = normalize(-mv.xyz);
  vLocal = position;
  gl_Position = projectionMatrix * mv;
}`;

const frag = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uFill;
uniform float uRim;
uniform vec3 uSize;
uniform vec4 uBand;   // y position (local), half height, intensity, unused
varying vec3 vN;
varying vec3 vView;
varying vec3 vLocal;
void main() {
  float f = pow(1.0 - abs(dot(normalize(vN), normalize(vView))), 2.5);
  // subtle vertical sheen so large faces aren't flat
  float sheen = 0.5 + 0.5 * smoothstep(-0.5, 0.5, vLocal.y / uSize.y);
  float band = uBand.z * (1.0 - smoothstep(uBand.y * 0.6, uBand.y, abs(vLocal.y - uBand.x)));
  float a = (uFill * (0.6 + 0.4 * sheen) + uRim * f + band * 0.35) * uOpacity;
  vec3 c = uColor * (1.0 + band * 2.0);
  gl_FragColor = vec4(c * a, a);
}`;

export interface GlassHandle {
  group: THREE.Group | null;
  setOpacity(a: number): void;
  /** highlight a horizontal band (local y, half-height, intensity) */
  setBand(y: number, half: number, k: number): void;
}

export interface GlassProps {
  size: [number, number, number];
  color: string;
  position?: [number, number, number];
  rotation?: [number, number, number];
  fill?: number;
  rim?: number;
  edge?: number;
}

export const Glass = forwardRef<GlassHandle, GlassProps>(function Glass(
  { size, color, position, rotation, fill = 0.05, rim = 0.35, edge = 0.55 },
  ref,
) {
  const group = useRef<THREE.Group>(null);
  const [w, h, d] = size;
  const geo = useMemo(() => new THREE.BoxGeometry(w, h, d), [w, h, d]);
  const edges = useMemo(() => new THREE.EdgesGeometry(geo), [geo]);
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: vert,
        fragmentShader: frag,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneMinusSrcAlphaFactor,
        uniforms: {
          uColor: { value: new THREE.Color(color) },
          uOpacity: { value: 1 },
          uFill: { value: fill },
          uRim: { value: rim },
          uSize: { value: new THREE.Vector3(w, h, d) },
          uBand: { value: new THREE.Vector4(0, 0, 0, 0) },
        },
      }),
    [color, fill, rim, w, h, d],
  );
  const lineMat = useMemo(
    () => new THREE.LineBasicMaterial({ color: new THREE.Color(color), transparent: true, opacity: edge, depthWrite: false }),
    [color, edge],
  );
  useImperativeHandle(ref, () => ({
    group: group.current,
    setOpacity(a) {
      mat.uniforms.uOpacity.value = a;
      lineMat.opacity = edge * a;
      if (group.current) group.current.visible = a > 0.002;
    },
    setBand(y, half, k) {
      (mat.uniforms.uBand.value as THREE.Vector4).set(y, half, k, 0);
    },
  }));
  return (
    <group ref={group} position={position} rotation={rotation}>
      <mesh geometry={geo} material={mat} renderOrder={1} />
      <lineSegments geometry={edges} material={lineMat} renderOrder={2} />
    </group>
  );
});

/** Thin line segments (brackets, hairlines). Points are pairs [a, b, a, b, …]. */
export const Lines = forwardRef<THREE.LineSegments, { points: number[]; color: string; opacity?: number }>(function Lines(
  { points, color, opacity = 0.8 },
  ref,
) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    return g;
  }, [points]);
  const mat = useMemo(() => new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }), [color, opacity]);
  return <lineSegments ref={ref} geometry={geo} material={mat} renderOrder={5} />;
});

/** Square brackets around a box [x0, x1] × [y0, y1] at depth z, as line-segment pairs. */
export function bracketPoints(x0: number, x1: number, y0: number, y1: number, z: number, lip = 0.14): number[] {
  return [
    x0, y0, z, x0, y1, z, x0, y1, z, x0 + lip, y1, z, x0, y0, z, x0 + lip, y0, z,
    x1, y0, z, x1, y1, z, x1, y1, z, x1 - lip, y1, z, x1, y0, z, x1 - lip, y0, z,
  ];
}
