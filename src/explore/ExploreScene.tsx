// Explore mode scene: the whole tower with orbit controls. Clicking a block focuses that layer
// (the camera flies to it); the residual stream at the focused depth is shown as real columns.

import { OrbitControls } from '@react-three/drei';
import { useFrame, useThree } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitImpl } from 'three-stdlib';
import { useApp } from '../state/store';
import { Columns, type ColumnsHandle } from '../scene/primitives/Columns';
import { Label, fade } from '../scene/primitives/Label';
import { TOWER, Tower, type TowerHandle } from '../scene/primitives/Tower';
import { TokenHeader, type TokenHeaderHandle } from '../scene/primitives/TokenHeader';
import { C, colX } from '../scene/theme';

/** z where the stream is displayed for depth d (0 = embeddings, d = after block d). */
export const depthZ = (d: number) => (d === 0 ? 6 : TOWER.mlpZ(d - 1) - 2.2);

export function ExploreScene() {
  const trace = useApp((s) => s.trace)!;
  const focus = useApp((s) => s.focus);
  const T = trace.T;
  const xs = useMemo(() => trace.tokens.map((_, k) => colX(k, T)), [trace, T]);
  const tower = useRef<TowerHandle>(null);
  const cols = useRef<ColumnsHandle>(null);
  const header = useRef<TokenHeaderHandle>(null);
  const label = useRef<HTMLDivElement>(null);
  const controls = useRef<OrbitImpl>(null);
  const camera = useThree((s) => s.camera);
  const fly = useRef<{ pos: THREE.Vector3; target: THREE.Vector3; t: number } | null>(null);
  const depth = focus.layer; // 0..12
  const z = depthZ(depth);

  useEffect(() => {
    tower.current?.setOpacity(1);
    tower.current?.setReach(TOWER.headZ);
    tower.current?.setBlocksOpacity(0.9);
    tower.current?.setPatternsOpacity(0.85);
    tower.current?.setLabelsOpacity(1);
    header.current?.setOpacity(1);
    header.current?.setIdOpacity(1);
    header.current?.setLineOpacity(0);
    fade(label.current, 1);
  });
  const first = useRef(true);
  useEffect(() => {
    cols.current?.setOpacity(1);
    if (first.current) {
      // opening view: the whole tower, three-quarter angle
      first.current = false;
      camera.position.set(36, 24, 20);
      controls.current?.target.set(-2, 0, -70);
      controls.current?.update();
      return;
    }
    // fly to the focused depth, keeping the current viewing direction
    const target = new THREE.Vector3(0, 0, z - 1);
    const dir = camera.position.clone().sub(controls.current?.target ?? new THREE.Vector3()).normalize();
    if (dir.lengthSq() < 0.5) dir.set(0.55, 0.35, 0.75).normalize();
    fly.current = { pos: target.clone().add(dir.multiplyScalar(24)), target, t: 0 };
  }, [z, camera, trace]);

  useFrame((_, dt) => {
    const f = fly.current;
    const c = controls.current;
    if (!f || !c) return;
    f.t = Math.min(1, f.t + dt * 1.2);
    const k = 1 - Math.pow(1 - f.t, 3);
    camera.position.lerp(f.pos, k * 0.18);
    c.target.lerp(f.target, k * 0.18);
    c.update();
    if (f.t >= 1 && camera.position.distanceTo(f.pos) < 0.05) fly.current = null;
  });

  return (
    <>
      <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.08} minDistance={4} maxDistance={260} onStart={() => (fly.current = null)} />
      <Tower ref={tower} trace={trace} />
      <TokenHeader ref={header} trace={trace} y={2.9} idY={2.3} />
      {Array.from({ length: 12 }, (_, L) => (
        <mesh
          key={L}
          position={[0, 0.6, TOWER.blockZ(L)]}
          onClick={(e) => {
            e.stopPropagation();
            useApp.getState().setFocus({ layer: L + 1 });
          }}
          onPointerOver={() => (document.body.style.cursor = 'pointer')}
          onPointerOut={() => (document.body.style.cursor = '')}
        >
          <boxGeometry args={[T * 2 + 1.2, 3.4, 10]} />
          <meshBasicMaterial visible={false} />
        </mesh>
      ))}
      <group position={[0, 0, z]}>
        <Columns key={`${depth}-${trace.prompt}`} ref={cols} data={trace.resid[depth]} D={768} xs={xs} y0={-1.2} color={C.token} />
      </group>
      <Label ref={label} position={[xs[0] - 1.4, -2.2, z]} tex={depth === 0 ? 'x^{0}' : `x^{${depth}} \\ (\\text{after block } ${depth})`} size={16} align="right" />
    </>
  );
}
