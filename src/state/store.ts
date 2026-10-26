import { create } from 'zustand';
import type { SceneTrace } from '../model/trace';
import type { SampleParams } from '../model/sample';

export type Mode = 'walk' | 'explore';
export type EngineStatus =
  | { kind: 'idle' }
  | { kind: 'loading'; loaded: number; total: number }
  | { kind: 'ready'; backend: string }
  | { kind: 'running' }
  | { kind: 'error'; message: string };

interface AppState {
  mode: Mode;
  trace: SceneTrace | null;
  engine: EngineStatus;
  sampling: SampleParams;
  reducedMotion: boolean;
  /** explore focus */
  focus: { layer: number; head: number; token: number | null };
  setMode(m: Mode): void;
  setTrace(t: SceneTrace): void;
  setEngine(e: EngineStatus): void;
  setSampling(p: Partial<SampleParams>): void;
  setReducedMotion(r: boolean): void;
  setFocus(f: Partial<AppState['focus']>): void;
}

const prefersReduced = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

export const useApp = create<AppState>((set) => ({
  mode: 'walk',
  trace: null,
  engine: { kind: 'idle' },
  sampling: { temperature: 1, topK: 40, topP: 0.9 },
  reducedMotion: !!prefersReduced,
  focus: { layer: 0, head: 0, token: null },
  setMode: (mode) => set({ mode }),
  setTrace: (trace) => set({ trace }),
  setEngine: (engine) => set({ engine }),
  setSampling: (p) => set((s) => ({ sampling: { ...s.sampling, ...p } })),
  setReducedMotion: (reducedMotion) => set({ reducedMotion }),
  setFocus: (f) => set((s) => ({ focus: { ...s.focus, ...f } })),
}));

/**
 * Story progress lives outside React so the render loop can read it every frame without
 * re-rendering: `target` comes from scroll, `s` is the damped value scenes animate with.
 */
export const story = { target: 0, s: 0, velocity: 0, frozen: false };
