// Any unicode string rendered once to a canvas texture and shown on a plane sized in world units.
// Used for token labels (which can contain characters outside the glyph atlas).

import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import * as THREE from 'three';
import katexMain from 'katex/dist/fonts/KaTeX_Main-Regular.ttf?url';

const PX = 128;
const FONT = `${PX}px UnfoldedMain, 'Latin Modern Roman', 'Times New Roman', serif`;
const cache = new Map<string, { tex: THREE.Texture; aspect: number }>();

let fontReady = false;
const face = typeof FontFace !== 'undefined' ? new FontFace('UnfoldedMain', `url(${katexMain})`) : null;
face?.load().then((f) => {
  document.fonts.add(f);
  fontReady = true;
  cache.clear();
});

/** Show leading spaces as a visible middle dot so " Jordan" reads as a different token than "Jordan". */
export const visibleToken = (s: string) => s.replace(/^ /, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥');

function textTexture(text: string) {
  const key = text + (fontReady ? '' : '#');
  const hit = cache.get(key);
  if (hit) return hit;
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d')!;
  ctx.font = FONT;
  const w = Math.ceil(ctx.measureText(text).width) + PX * 0.3;
  c.width = Math.max(8, w);
  c.height = Math.round(PX * 1.4);
  ctx.font = FONT;
  ctx.fillStyle = 'white';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  // leading-space dot drawn dimmer
  if (text.startsWith('·')) {
    const rest = text.slice(1);
    const dotW = ctx.measureText('·').width;
    const total = ctx.measureText(text).width;
    const x0 = c.width / 2 - total / 2;
    ctx.globalAlpha = 0.22;
    ctx.textAlign = 'left';
    ctx.fillText('·', x0, c.height / 2);
    ctx.globalAlpha = 1;
    ctx.fillText(rest, x0 + dotW, c.height / 2);
  } else ctx.fillText(text, c.width / 2, c.height / 2);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const out = { tex, aspect: c.width / c.height };
  cache.set(key, out);
  return out;
}

/** Width in world units of a label of the given height (matches TextPlane's glyph extent). */
export function measureText(text: string, height: number): number {
  const c = document.createElement('canvas').getContext('2d')!;
  c.font = FONT;
  return (c.measureText(text).width / (PX * 1.4)) * height * 1.4;
}

export interface TextPlaneHandle {
  group: THREE.Group | null;
  setOpacity(a: number): void;
  setColor(c: THREE.ColorRepresentation, k?: number): void;
}

export const TextPlane = forwardRef<
  TextPlaneHandle,
  { text: string; height: number; color?: string; position?: [number, number, number]; anchor?: 'center' | 'left' | 'right' }
>(function TextPlane({ text, height, color = '#f4eee2', position, anchor = 'center' }, ref) {
  const { tex, aspect } = useMemo(() => textTexture(text), [text]);
  const h = height * 1.4;
  const w = h * aspect;
  const group = useRef<THREE.Group>(null);
  const mat = useMemo(
    () => new THREE.MeshBasicMaterial({ map: tex, color: new THREE.Color(color), transparent: true, depthWrite: false, toneMapped: false, opacity: 0 }),
    [tex, color],
  );
  useImperativeHandle(ref, () => ({
    group: group.current,
    setOpacity(a) {
      mat.opacity = a;
      if (group.current) group.current.visible = a > 0.002;
    },
    setColor(c, k = 1) {
      mat.color.set(c).multiplyScalar(k);
    },
  }));
  const ox = anchor === 'left' ? w / 2 : anchor === 'right' ? -w / 2 : 0;
  return (
    <group ref={group} position={position}>
      <mesh position={[ox, 0, 0]} material={mat} renderOrder={12}>
        <planeGeometry args={[w, h]} />
      </mesh>
    </group>
  );
});
