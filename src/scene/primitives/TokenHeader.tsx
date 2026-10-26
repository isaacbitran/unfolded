// Token labels floating above their columns, a hairline dropping into each column, and the
// integer id under each label.

import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { GlyphField, layoutText, useAtlas, type GlyphFieldHandle } from '../glyphs/GlyphField';
import { C, colX } from '../theme';
import { Lines } from './Glass';
import { TextPlane, visibleToken, type TextPlaneHandle } from './TextPlane';

export interface TokenHeaderHandle {
  setOpacity(a: number): void;
  setIdOpacity(a: number): void;
  setLineOpacity(a: number): void;
  /** emphasise one token (others dim); null = all equal */
  setFocus(k: number | null, amount: number): void;
}

export const TokenHeader = forwardRef<
  TokenHeaderHandle,
  { trace: SceneTrace; y: number; idY: number; lineTo?: number; z?: number; height?: number }
>(function TokenHeader({ trace, y, idY, lineTo = 1.55, z = 0, height = 0.36 }, ref) {
  const atlas = useAtlas();
  const T = trace.T;
  const planes = useRef<(TextPlaneHandle | null)[]>([]);
  const ids = useRef<GlyphFieldHandle>(null);
  const lines = useRef<THREE.LineSegments>(null);
  const state = useRef({ a: 0, focus: null as number | null, amount: 0 });

  const idItems = useMemo(
    () =>
      atlas
        ? trace.tokens.flatMap((t, k) => layoutText(String(t.id), colX(k, T), idY, z, { h: 0.25, color: new THREE.Color(C.dim), k: 1.1 }))
        : [],
    [atlas, trace, T, idY, z],
  );
  const linePts = useMemo(() => trace.tokens.flatMap((_, k) => [colX(k, T), idY - 0.2, z, colX(k, T), lineTo, z]), [trace, T, idY, lineTo, z]);

  const apply = () => {
    const { a, focus, amount } = state.current;
    planes.current.forEach((p, k) => {
      const dim = focus === null || focus === k ? 1 : 1 - 0.75 * amount;
      p?.setOpacity(a * dim);
    });
  };

  useImperativeHandle(ref, () => ({
    setOpacity(a) {
      state.current.a = a;
      apply();
    },
    setIdOpacity(a) {
      ids.current?.setOpacity(a);
    },
    setLineOpacity(a) {
      if (!lines.current) return;
      (lines.current.material as THREE.LineBasicMaterial).opacity = 0.3 * a;
      lines.current.visible = a > 0.002;
    },
    setFocus(k, amount) {
      state.current.focus = k;
      state.current.amount = amount;
      apply();
    },
  }));

  return (
    <group>
      {trace.tokens.map((t, k) => (
        <TextPlane key={k} ref={(el) => void (planes.current[k] = el)} text={visibleToken(t.text)} height={height} position={[colX(k, T), y, z]} />
      ))}
      <GlyphField ref={ids} items={idItems} />
      <Lines ref={lines} points={linePts} color={C.token} opacity={0} />
    </group>
  );
});
