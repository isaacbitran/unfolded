import { Canvas } from '@react-three/fiber';
import { Bloom, EffectComposer, SMAA, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import { CameraRig } from './CameraRig';
import { World } from './World';
import { PerfProbe } from './PerfProbe';
import { ExploreScene } from '../explore/ExploreScene';
import { useApp } from '../state/store';

export function Stage() {
  const mode = useApp((s) => s.mode);
  const trace = useApp((s) => s.trace);
  return (
    <Canvas
      className="stage"
      dpr={[1, 2]}
      gl={{ antialias: false, powerPreference: 'high-performance', alpha: false, preserveDrawingBuffer: true }}
      camera={{ fov: 34, near: 0.1, far: 1000, position: [0, 0, 20] }}
      onCreated={({ gl }) => {
        gl.setClearColor('#000000');
        gl.toneMapping = THREE.NoToneMapping;
      }}
    >
      <CameraRig enabled={mode === 'walk'} />
      {mode === 'walk' ? <World /> : trace && <ExploreScene />}
      <PerfProbe />
      <EffectComposer multisampling={4}>
        <Bloom mipmapBlur luminanceThreshold={0.92} luminanceSmoothing={0.15} intensity={0.85} radius={0.7} />
        <Vignette offset={0.25} darkness={0.75} />
        <SMAA />
      </EffectComposer>
    </Canvas>
  );
}
