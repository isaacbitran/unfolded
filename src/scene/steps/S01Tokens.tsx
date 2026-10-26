// Step 1 — tokenization. The prompt sits as one sentence; it splits at token boundaries, each
// piece gets a glass tile and slides to its column; the integer ids fade in underneath.

import { useMemo, useRef } from 'react';
import type * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { Glass, type GlassHandle } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TextPlane, measureText, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { TokenHeader, type TokenHeaderHandle } from '../primitives/TokenHeader';
import { C, colX, easeInOut, lerp, seg } from '../theme';
import { useStep } from '../useStep';
import { TOKEN_ID_Y, TOKEN_LABEL_Y } from './S02Embedding';

const H = 0.36;

export function S01Tokens({ index, trace }: { index: number; trace: SceneTrace }) {
  const T = trace.T;
  const planes = useRef<(TextPlaneHandle | null)[]>([]);
  const tiles = useRef<(GlassHandle | null)[]>([]);
  const header = useRef<TokenHeaderHandle>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const groups = useRef<(THREE.Group | null)[]>([]);

  // sentence layout: pieces touching, centered
  const layout = useMemo(() => {
    const widths = trace.tokens.map((t) => measureText(visibleToken(t.text), H));
    const total = widths.reduce((a, b) => a + b, 0);
    let x = -total / 2;
    return widths.map((w, k) => {
      const sx = x + w / 2;
      x += w;
      return { w, sx, ex: colX(k, T) };
    });
  }, [trace, T]);

  useStep(index, (p, vis) => {
    const split = easeInOut(seg(p, 0.18, 0.55));
    layout.forEach((l, k) => {
      const stagger = seg(p, 0.18 + k * 0.025, 0.5 + k * 0.025);
      const t = easeInOut(stagger);
      const g = groups.current[k];
      if (g) g.position.x = lerp(l.sx, l.ex, t);
      planes.current[k]?.setOpacity(vis);
      tiles.current[k]?.setOpacity(vis * split * (1 - 0.6 * seg(p, 0.85, 1)));
    });
    header.current?.setOpacity(0);
    header.current?.setIdOpacity(vis * seg(p, 0.55, 0.75));
    fade(labels.current[0], vis * seg(p, 0.6, 0.72) * (1 - seg(p, 0.92, 1)));
    fade(labels.current[1], vis * (1 - seg(p, 0.12, 0.2)));
  });

  return (
    <group>
      {trace.tokens.map((t, k) => (
        <group key={k} ref={(g) => void (groups.current[k] = g)} position={[layout[k]?.sx ?? 0, TOKEN_LABEL_Y, 0]}>
          <TextPlane ref={(el) => void (planes.current[k] = el)} text={visibleToken(t.text)} height={H} />
          <Glass ref={(el) => void (tiles.current[k] = el)} size={[(layout[k]?.w ?? 1) + 0.35, 0.62, 0.08]} position={[0, 0.02, -0.06]} color={C.token} fill={0.003} rim={0.18} edge={0.35} />
        </group>
      ))}
      <TokenHeader ref={header} trace={trace} y={TOKEN_LABEL_Y} idY={TOKEN_ID_Y} />
      <Label ref={(el) => void (labels.current[0] = el)} position={[0, TOKEN_ID_Y - 0.65, 0]} tex={`t_i \\in \\{0, \\dots, 50256\\} \\qquad T = ${T}`} size={18} color="#b9bcc2" />
      <Label ref={(el) => void (labels.current[1] = el)} position={[0, TOKEN_LABEL_Y + 0.75, 0]} text="your prompt" size={13} color="#7d828b" />
    </group>
  );
}
