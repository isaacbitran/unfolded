// DOM labels pinned to 3D anchors: KaTeX math and serif text, crisp at any DPI.
// Each label is a plain div in one overlay layer, positioned every frame by projecting its anchor
// (no React roots per label, so mounting dozens is cheap and StrictMode-safe).

import { useFrame, useThree } from '@react-three/fiber';
import katex from 'katex';
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from 'react';
import * as THREE from 'three';

let layer: HTMLDivElement | null = null;
function overlay(): HTMLDivElement {
  if (!layer) {
    layer = document.createElement('div');
    layer.className = 'label-layer';
    document.body.appendChild(layer);
  }
  return layer;
}

export interface LabelProps {
  position: [number, number, number];
  /** KaTeX source (math) or plain text */
  tex?: string;
  text?: string;
  size?: number; // px
  color?: string;
  align?: 'center' | 'left' | 'right';
  className?: string;
}

const v = new THREE.Vector3();

export const Label = forwardRef<HTMLDivElement | null, LabelProps>(function Label(
  { position, tex, text, size = 18, color = '#f4eee2', align = 'center', className },
  ref,
) {
  const anchor = useRef<THREE.Group>(null);
  const el = useMemo(() => {
    const d = document.createElement('div');
    d.className = 'label3d';
    d.style.opacity = '0';
    d.style.visibility = 'hidden';
    return d;
  }, []);
  useImperativeHandle(ref, () => el, [el]);
  useEffect(() => {
    overlay().appendChild(el);
    return () => el.remove();
  }, [el]);
  useEffect(() => {
    el.className = `label3d ${className ?? ''}`;
    el.style.fontSize = `${size * (window.innerWidth < 760 ? 0.62 : 1)}px`;
    el.style.color = color;
    if (tex) el.innerHTML = katex.renderToString(tex, { throwOnError: false, output: 'html' });
    else el.textContent = text ?? '';
  }, [el, tex, text, size, color, className]);
  const camera = useThree((s) => s.camera);
  const sizeRef = useThree((s) => s.size);
  const tx = align === 'center' ? -50 : align === 'left' ? 0 : -100;
  const place = () => {
    if (!anchor.current) return;
    anchor.current.getWorldPosition(v);
    v.project(camera);
    const behind = v.z > 1;
    const x = (v.x * 0.5 + 0.5) * sizeRef.width;
    const y = (-v.y * 0.5 + 0.5) * sizeRef.height;
    el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(${tx}%, -50%)`;
    el.style.display = behind ? 'none' : '';
  };
  (el as LabelEl).__place = place;
  useFrame(() => {
    if (el.style.visibility !== 'hidden') place();
  });
  return <group ref={anchor} position={position} />;
});

/** Set opacity on a label element (no React re-render). */
export const fade = (el: HTMLDivElement | null, a: number) => {
  if (!el) return;
  const o = a < 0.003 ? '0' : a.toFixed(3);
  if (el.style.opacity !== o) el.style.opacity = o;
  const vis = a < 0.003 ? 'hidden' : 'visible';
  if (el.style.visibility !== vis) {
    el.style.visibility = vis;
    if (vis === 'visible') (el as LabelEl).__place?.();
  }
};

type LabelEl = HTMLDivElement & { __place?: () => void };
