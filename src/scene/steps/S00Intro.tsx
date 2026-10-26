// Step 0 — cold open. The whole 12-block tower in silhouette, the prompt at the front, and the
// model's real top prediction waiting at the far end.

import { useRef } from 'react';
import type { SceneTrace } from '../../model/trace';
import { Label, fade } from '../primitives/Label';
import { TOWER, Tower, type TowerHandle } from '../primitives/Tower';
import { TokenHeader, type TokenHeaderHandle } from '../primitives/TokenHeader';
import { TextPlane, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { seg } from '../theme';
import { useStep } from '../useStep';

export function S00Intro({ index, trace }: { index: number; trace: SceneTrace }) {
  const tower = useRef<TowerHandle>(null);
  const header = useRef<TokenHeaderHandle>(null);
  const answer = useRef<TextPlaneHandle>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const top = trace.positions[trace.T - 1].top[0];
  useStep(index, (p, vis) => {
    tower.current?.setOpacity(vis);
    tower.current?.setReach(TOWER.headZ);
    tower.current?.setBlocksOpacity(vis * 0.8);
    tower.current?.setPatternsOpacity(vis * 0.6);
    tower.current?.setLabelsOpacity(0);
    tower.current?.setPulse(2 + (TOWER.headZ - 2) * seg(p, 0.1, 0.8), vis * 0.9);
    header.current?.setOpacity(vis * seg(p, 0.0, 0.15));
    header.current?.setIdOpacity(0);
    answer.current?.setOpacity(vis * seg(p, 0.7, 0.85));
    fade(labels.current[0], vis * seg(p, 0.75, 0.9));
  });
  return (
    <group>
      <Tower ref={tower} trace={trace} />
      <TokenHeader ref={header} trace={trace} y={1.6} idY={-50} />
      <TextPlane ref={answer} text={visibleToken(top.text)} height={1.1} position={[0, 2.2, TOWER.headZ]} />
      <Label ref={(e) => void (labels.current[0] = e)} position={[0, 0.9, TOWER.headZ]} tex={`p = ${(top.p * 100).toFixed(1)}\\%`} size={16} color="#b9bcc2" />
    </group>
  );
}
