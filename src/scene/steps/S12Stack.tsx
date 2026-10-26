// Step 12 — many repetitions. The camera pulls back along the tower; every block shows its real
// attention patterns (12 heads) and MLP firing rate; a pulse runs from the embeddings to the head.

import { useMemo, useRef } from 'react';
import type { SceneTrace } from '../../model/trace';
import { Label, fade } from '../primitives/Label';
import { TOWER, Tower, firingRates, type TowerHandle } from '../primitives/Tower';
import { TokenHeader, type TokenHeaderHandle } from '../primitives/TokenHeader';
import { colX, lerp, seg } from '../theme';
import { useStep } from '../useStep';

export function S12Stack({ index, trace }: { index: number; trace: SceneTrace }) {
  const T = trace.T;
  const tower = useRef<TowerHandle>(null);
  const header = useRef<TokenHeaderHandle>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const rates = useMemo(() => firingRates(trace), [trace]);
  useStep(index, (p, vis) => {
    tower.current?.setOpacity(vis);
    tower.current?.setReach(TOWER.headZ);
    tower.current?.setBlocksOpacity(vis);
    tower.current?.setPatternsOpacity(vis * seg(p, 0.05, 0.25));
    tower.current?.setLabelsOpacity(vis * seg(p, 0.1, 0.3) * (1 - seg(p, 0.75, 0.9)));
    const pulse = (p * 1.6) % 1;
    tower.current?.setPulse(lerp(2, TOWER.headZ, pulse), vis * 1.1);
    header.current?.setOpacity(vis);
    header.current?.setIdOpacity(0);
    fade(labels.current[0], vis * seg(p, 0.35, 0.5));
    for (let L = 0; L < 12; L++) fade(labels.current[1 + L], vis * seg(p, 0.15 + L * 0.02, 0.3 + L * 0.02) * (1 - seg(p, 0.75, 0.9)));
  });
  const right = colX(T - 1, T) + 2.4;
  return (
    <group>
      <Tower ref={tower} trace={trace} />
      <TokenHeader ref={header} trace={trace} y={2.6} idY={-50} />
      <Label ref={(e) => void (labels.current[0] = e)} position={[0, 6, -100]} tex="\text{Many repetitions: } 12 \times (\text{Attn} + \text{MLP})" size={22} />
      {rates.map((r, L) => (
        <Label key={L} ref={(e) => void (labels.current[1 + L] = e)} position={[right, -1.2, TOWER.mlpZ(L)]} text={`MLP ${(r * 100).toFixed(0)}% on`} size={11} color="#ffb454" align="left" />
      ))}
    </group>
  );
}
