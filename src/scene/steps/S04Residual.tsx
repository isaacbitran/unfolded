// Step 4 — the residual stream. Each token's column stretches into a rail that runs through all 12
// blocks; a pulse travels down the rails; rail brightness follows the real norm ‖x‖ at each depth.

import { useMemo, useRef } from 'react';
import type { SceneTrace } from '../../model/trace';
import { Columns, type ColumnsHandle } from '../primitives/Columns';
import { Label, fade } from '../primitives/Label';
import { TOWER, Tower, streamNorms, type TowerHandle } from '../primitives/Tower';
import { TokenHeader, type TokenHeaderHandle } from '../primitives/TokenHeader';
import { C, colX, easeInOut, lerp, seg } from '../theme';
import { useStep } from '../useStep';
import { COL_TOP, TOKEN_ID_Y, TOKEN_LABEL_Y } from './S02Embedding';

export function S04Residual({ index, trace }: { index: number; trace: SceneTrace }) {
  const T = trace.T;
  const xs = useMemo(() => trace.tokens.map((_, k) => colX(k, T)), [trace, T]);
  const tower = useRef<TowerHandle>(null);
  const cols = useRef<ColumnsHandle>(null);
  const header = useRef<TokenHeaderHandle>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const norms = useMemo(() => streamNorms(trace), [trace]);
  const last = norms.map((n) => n[T - 1]);
  const normTex = `\\lVert x_{${T}}\\rVert:\\ ${last[0].toFixed(1)} \\to ${last[4].toFixed(1)} \\to ${last[8].toFixed(1)} \\to ${last[12].toFixed(1)}`;

  useStep(index, (p, vis) => {
    cols.current?.setOpacity(vis * (1 - seg(p, 0.08, 0.22)));
    header.current?.setOpacity(vis * (1 - seg(p, 0.3, 0.5) * 0.4));
    header.current?.setIdOpacity(0);
    const grow = easeInOut(seg(p, 0.1, 0.55));
    tower.current?.setOpacity(vis * seg(p, 0.05, 0.12));
    tower.current?.setReach(lerp(2, TOWER.headZ, grow));
    tower.current?.setBlocksOpacity(vis * seg(p, 0.2, 0.5));
    tower.current?.setPatternsOpacity(vis * 0.5 * seg(p, 0.45, 0.7));
    tower.current?.setLabelsOpacity(vis * seg(p, 0.4, 0.55));
    const pulse = seg(p, 0.5, 0.97);
    tower.current?.setPulse(lerp(2, TOWER.headZ, pulse), vis * Math.sin(Math.PI * pulse) * 1.2);
    fade(labels.current[0], vis * seg(p, 0.3, 0.42));
    fade(labels.current[1], vis * seg(p, 0.6, 0.72));
    fade(labels.current[2], vis * seg(p, 0.75, 0.85));
  });

  return (
    <group>
      <Tower ref={tower} trace={trace} />
      <Columns ref={cols} data={trace.resid[0]} D={768} xs={xs} y0={COL_TOP} color={C.token} />
      <TokenHeader ref={header} trace={trace} y={TOKEN_LABEL_Y} idY={TOKEN_ID_Y} />
      <Label ref={(e) => void (labels.current[0] = e)} position={[xs[T - 1] + 2.2, 0, -7]} tex={`\\text{residual stream } x \\in \\mathbb{R}^{${T}\\times768}`} size={16} align="left" />
      <Label ref={(e) => void (labels.current[1] = e)} position={[xs[T - 1] + 2.2, 0, -40]} tex="x \leftarrow x + \text{block}(x)" size={16} color="#d8d4ca" align="left" />
      <Label ref={(e) => void (labels.current[2] = e)} position={[xs[T - 1] + 2.2, -1.2, -90]} tex={normTex} size={14} color="#a9adb5" align="left" />
    </group>
  );
}
