import { useFrame, useThree } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useApp, story } from '../state/store';
import { cameraAt } from '../story/camera';

/** Damps story progress toward the scroll target and drives the camera along the story path. */
export function CameraRig({ enabled = true }: { enabled?: boolean }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const size = useThree((s) => s.size);
  const mode = useApp((s) => s.mode);
  // Lens shift: in the walkthrough the text column covers the left of the screen, so the
  // optical center moves right by half the column's width (labels follow: they use the same matrix).
  useEffect(() => {
    const shift = mode === 'walk' && size.width > 900 ? Math.min(420, size.width * 0.34 + 28) / 2 : 0;
    if (shift) camera.setViewOffset(size.width, size.height, -shift, 0, size.width, size.height);
    else camera.clearViewOffset();
    camera.updateProjectionMatrix();
  }, [camera, size, mode]);
  const reduced = useApp((s) => s.reducedMotion);
  const look = useRef(new THREE.Vector3());
  const pointer = useRef({ x: 0, y: 0 });
  useFrame((state, dt) => {
    dt = Math.min(dt, 1 / 20);
    if (story.frozen) story.s = story.target;
    else if (reduced) story.s += (story.target - story.s) * Math.min(1, dt * 12);
    else {
      // critically damped spring toward the target
      const w = 3.2;
      const x = story.s - story.target;
      const a = -w * w * x - 2 * w * story.velocity;
      story.velocity += a * dt;
      story.s += story.velocity * dt;
      if (Math.abs(x) < 1e-4 && Math.abs(story.velocity) < 1e-4) story.s = story.target;
    }
    if (!enabled) return;
    const pose = cameraAt(story.s);
    // gentle parallax from the pointer so the scene feels like a space, not a slide
    const px = reduced || story.frozen ? 0 : state.pointer.x;
    const py = reduced || story.frozen ? 0 : state.pointer.y;
    pointer.current.x += (px - pointer.current.x) * Math.min(1, dt * 2);
    pointer.current.y += (py - pointer.current.y) * Math.min(1, dt * 2);
    camera.position.set(pose.pos[0] + pointer.current.x * 0.5, pose.pos[1] + pointer.current.y * 0.3, pose.pos[2]);
    look.current.set(pose.look[0], pose.look[1], pose.look[2]);
    camera.lookAt(look.current);
    if (Math.abs(camera.fov - pose.fov) > 1e-3) {
      camera.fov = pose.fov;
      camera.updateProjectionMatrix();
    }
  });
  return null;
}
