// Step 15 — autoregression and the KV cache. The sampled token joins the input as a new column;
// only its rail runs through the 12 blocks. At every block the earlier tokens' cached keys (red)
// and values (green) light up as they are read, and the new token's pair is written next to them.

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { Lines } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TOWER, Tower, type TowerHandle } from '../primitives/Tower';
import { TextPlane, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { C, colX, lerp, seg } from '../theme';
import { useStep } from '../useStep';
import { walkthroughSample } from './S14Sampling';

const SQ = 0.36;

export function S15Autoregress({ index, trace }: { index: number; trace: SceneTrace }) {
  const T = trace.T;
  const S = T + 1;
  const next = useMemo(() => walkthroughSample(trace), [trace]);
  const tower = useRef<TowerHandle>(null);
  const word = useRef<TextPlaneHandle>(null);
  const oldWords = useRef<(TextPlaneHandle | null)[]>([]);
  const rail = useRef<THREE.LineSegments>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const xNew = colX(T, S);

  // cache cells: per block, per token, a key square and a value square above the attention slab
  const cache = useMemo(() => {
    const n = 12 * S * 2;
    const mesh = new THREE.InstancedMesh(
      new THREE.PlaneGeometry(SQ, SQ),
      new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false }),
      n,
    );
    const m = new THREE.Matrix4();
    let i = 0;
    for (let L = 0; L < 12; L++)
      for (let t = 0; t < S; t++)
        for (let kv = 0; kv < 2; kv++) {
          m.makeTranslation(colX(t, S) + (kv ? 0.22 : -0.22), TOWER.railY + 2.75, TOWER.attnZ(L));
          mesh.setMatrixAt(i++, m);
        }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.frustumCulled = false;
    mesh.renderOrder = 8;
    return mesh;
  }, [S]);
  const red = useMemo(() => new THREE.Color(C.key), []);
  const green = useMemo(() => new THREE.Color(C.value), []);
  const tmp = useMemo(() => new THREE.Color(), []);

  useStep(index, (p, vis) => {
    tower.current?.setOpacity(vis * 0.7);
    tower.current?.setReach(TOWER.headZ);
    tower.current?.setBlocksOpacity(vis * 0.55);
    tower.current?.setPatternsOpacity(vis * 0.2);
    tower.current?.setLabelsOpacity(vis * seg(p, 0.2, 0.3));
    tower.current?.setPulse(1e3, 0);
    // the new token arrives from the far end (where it was sampled) to the front
    const arrive = seg(p, 0.0, 0.36);
    word.current?.setOpacity(vis);
    if (word.current?.group) {
      word.current.group.position.set(lerp(0, xNew, arrive), lerp(2.2, 2.6, arrive), lerp(TOWER.headZ, 0, arrive * arrive * (3 - 2 * arrive)));
    }
    oldWords.current.forEach((w) => w?.setOpacity(vis * 0.9));
    // the new rail grows through the tower
    const z = lerp(2, TOWER.headZ, seg(p, 0.4, 0.86));
    if (rail.current) {
      rail.current.scale.z = Math.max(1e-3, (2 - z) / (2 - TOWER.headZ));
      (rail.current.material as THREE.LineBasicMaterial).opacity = vis * seg(p, 0.26, 0.3);
      rail.current.visible = vis > 0.002;
    }
    // cache: old K/V glow while being read (as the new column passes their block), new pair is written
    let i = 0;
    for (let L = 0; L < 12; L++) {
      const bz = TOWER.attnZ(L);
      const passing = Math.exp(-(((z - bz) / 3) ** 2));
      const written = z < bz ? 1 : 0;
      for (let t = 0; t < S; t++)
        for (let kv = 0; kv < 2; kv++) {
          const base = kv ? green : red;
          const isNew = t === T;
          const k = isNew ? written * (0.5 + 1.2 * passing) : seg(p, 0.2, 0.28) * (0.22 + 1.3 * passing);
          cache.setColorAt(i++, tmp.copy(base).multiplyScalar(k * vis));
        }
    }
    if (cache.instanceColor) cache.instanceColor.needsUpdate = true;
    cache.visible = vis > 0.002;
    fade(labels.current[0], vis * seg(p, 0.3, 0.4));
    fade(labels.current[1], vis * seg(p, 0.86, 0.95));
    fade(labels.current[2], vis * seg(p, 0.3, 0.4));
  });

  return (
    <group>
      <Tower ref={tower} trace={trace} slots={S} />
      {trace.tokens.map((t, k) => (
        <TextPlane key={k} ref={(e) => void (oldWords.current[k] = e)} text={visibleToken(t.text)} height={0.36} position={[colX(k, S), 2.6, 0]} />
      ))}
      <group>
        <TextPlane ref={word} text={visibleToken(next.text)} height={0.42} color="#ffffff" />
      </group>
      <group position={[xNew, TOWER.railY, 2]}>
        <Lines ref={rail} points={[0, 0, 0, 0, 0, TOWER.headZ - 2]} color="#ffffff" opacity={0} />
      </group>
      <primitive object={cache} />
      <Label ref={(e) => void (labels.current[0] = e)} position={[colX(0, S) - 1.4, TOWER.railY + 2.75, TOWER.attnZ(0)]} tex="\text{KV cache: } {\color{#ff6b6b}k}, {\color{#5fd68a}v}" size={14} align="right" />
      <Label ref={(e) => void (labels.current[1] = e)} position={[0, -2.2, 0]} tex={`\\text{computed this step: } 1 \\text{ column} \\quad\\cdot\\quad \\text{reused: } ${T} \\times 12 \\times (k, v)`} size={17} />
      <Label ref={(e) => void (labels.current[2] = e)} position={[xNew, 3.4, 0]} text="new token" size={12} color="#9a9ea6" />
    </group>
  );
}
