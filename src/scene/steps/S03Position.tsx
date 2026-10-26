// Step 3 — positional embeddings. Rows 0..T−1 of W_P (amber) fly out of their slab and line up
// under the token columns; then each pair merges and the real sum x⁰ = W_E[t] + W_P[i] takes its place.

import { useMemo, useRef } from 'react';
import type * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { Columns, type ColumnsHandle } from '../primitives/Columns';
import { Label, fade } from '../primitives/Label';
import { MatrixSlab, type MatrixSlabHandle } from '../primitives/MatrixSlab';
import { TokenHeader, type TokenHeaderHandle } from '../primitives/TokenHeader';
import { C, L, colX, easeInOut, seg } from '../theme';
import { useStep } from '../useStep';
import { columnBox } from '../vectors';
import { COL_TOP, TOKEN_ID_Y, TOKEN_LABEL_Y } from './S02Embedding';

const PSLAB = { x: 9.6, y: -0.6, w: 4.6, h: 6.0 };
const POS_TOP = COL_TOP - 3.55;

export function S03Position({ index, trace }: { index: number; trace: SceneTrace }) {
  const T = trace.T;
  const xs = useMemo(() => trace.tokens.map((_, k) => colX(k, T)), [trace, T]);
  const tok = useRef<ColumnsHandle>(null);
  const pos = useRef<ColumnsHandle>(null);
  const sum = useRef<ColumnsHandle>(null);
  const slab = useRef<MatrixSlabHandle>(null);
  const header = useRef<TokenHeaderHandle>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const posGroup = useRef<THREE.Group>(null);

  // amber columns start at the slab's face, row i of W_P
  const posFrom = useMemo(
    () => trace.tokens.map((_, i) => [PSLAB.x - 0.3, PSLAB.y + PSLAB.h / 2 - 0.6 - i * 0.55, 0.25] as [number, number, number]),
    [trace],
  );
  const posDelay = useMemo(() => (c: number, r: number) => [c * 0.06 + r * 0.004, 0.55 + c * 0.06 + r * 0.004] as [number, number], []);

  useStep(index, (p, vis) => {
    const slabA = vis * seg(p, 0, 0.12) * (1 - seg(p, 0.82, 0.95));
    slab.current?.setOpacity(slabA);
    header.current?.setOpacity(vis);
    header.current?.setIdOpacity(vis * (1 - seg(p, 0.05, 0.15)));
    // phase 1: position rows fly into place below each token column
    const fly = seg(p, 0.1, 0.5);
    pos.current?.setProgress(fly);
    pos.current?.setArc(1.4);
    // phase 2: merge
    const merge = easeInOut(seg(p, 0.55, 0.78));
    const tokOp = 1 - seg(p, 0.66, 0.8);
    tok.current?.setOpacity(vis * tokOp);
    pos.current?.setOpacity(vis * seg(p, 0.08, 0.14) * tokOp);
    pos.current?.setBracketOpacity(vis * seg(p, 0.46, 0.52) * (1 - seg(p, 0.55, 0.62)));
    tok.current?.setBracketOpacity(vis * tokOp * (1 - seg(p, 0.55, 0.62)));
    // the amber column rises onto the token column; the sum takes their place
    if (posGroup.current) posGroup.current.position.y = (COL_TOP - POS_TOP) * merge;
    sum.current?.setOpacity(vis * seg(p, 0.7, 0.82));
    fade(labels.current[0], vis * seg(p, 0.42, 0.5) * (1 - seg(p, 0.56, 0.62)));
    fade(labels.current[1], vis * seg(p, 0.8, 0.9));
    trace.tokens.forEach((_, k) => fade(labels.current[2 + k], vis * seg(p, 0.42, 0.5) * (1 - seg(p, 0.56, 0.62))));
  });

  const plusY = COL_TOP - (L.visible + 1) * L.rowH - 0.62;
  return (
    <group>
      <MatrixSlab ref={slab} preview={trace.previews.W_P} position={[PSLAB.x, PSLAB.y, 0]} w={PSLAB.w} h={PSLAB.h} color={C.position} name="W_P" />
      <Columns ref={tok} data={trace.embedTok} D={768} xs={xs} y0={COL_TOP} color={C.token} />
      <group ref={posGroup}>
        <Columns ref={pos} data={trace.embedPos} D={768} xs={xs} y0={POS_TOP} color={C.position} from={posFrom} delay={posDelay} />
      </group>
      <Columns ref={sum} data={trace.resid[0]} D={768} xs={xs} y0={COL_TOP} color={C.token} />
      <TokenHeader ref={header} trace={trace} y={TOKEN_LABEL_Y} idY={TOKEN_ID_Y} />
      <Label ref={(e) => void (labels.current[0] = e)} position={[xs[0] - 1.6, plusY, 0]} tex="W_P[i]" size={16} color={C.position} align="right" />
      <Label ref={(e) => void (labels.current[1] = e)} position={[0, columnBox(0, COL_TOP).y0 - 0.75, 0]} tex={`X^{0} = W_E[t] + W_P \\in \\mathbb{R}^{${T}\\times 768}`} size={20} />
      {xs.map((x, k) => (
        <Label key={k} ref={(e) => void (labels.current[2 + k] = e)} position={[x, plusY, 0]} tex="+" size={22} color="#d8d4ca" />
      ))}
    </group>
  );
}
