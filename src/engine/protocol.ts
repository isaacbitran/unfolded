import type { Candidate, SampleParams } from '../model/sample';
import type { SceneTrace } from '../model/trace';

export type ToWorker =
  | { type: 'load'; url: string; base: string; preferGPU: boolean }
  | { type: 'run'; prompt: string; ids: number[] }
  | { type: 'generate'; n: number; params: SampleParams; seed: number };

export type FromWorker =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'ready'; backend: string }
  | { type: 'trace'; trace: SceneTrace; ms: number }
  | { type: 'token'; id: number; text: string; u: number; candidates: (Candidate & { text: string })[]; ms: number; cached: number }
  | { type: 'error'; message: string };
