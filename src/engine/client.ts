// Main-thread side of the engine: owns the worker, the tokenizer for live tokenization, and pushes
// results into the app store.

import { GPT2Tokenizer } from '../model/tokenizer';
import type { FromWorker, ToWorker } from './protocol';
import { useApp } from '../state/store';

export const WEIGHTS_URL: string =
  import.meta.env.VITE_WEIGHTS_URL ?? (import.meta.env.DEV ? '/local-weights/model.safetensors' : 'https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors');
export const WEIGHTS_MB = 498;

let worker: Worker | null = null;
let loading: Promise<void> | null = null;
let tokenizer: Promise<GPT2Tokenizer> | null = null;
const listeners = new Set<(m: FromWorker) => void>();

export function getTokenizer(): Promise<GPT2Tokenizer> {
  if (!tokenizer) {
    const base = import.meta.env.BASE_URL;
    tokenizer = Promise.all([
      fetch(`${base}tokenizer/vocab.json`).then((r) => r.json()),
      fetch(`${base}tokenizer/merges.txt`).then((r) => r.text()),
    ]).then(([v, m]) => new GPT2Tokenizer(v, m));
  }
  return tokenizer;
}

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('../worker/engine.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (e: MessageEvent<FromWorker>) => {
    const m = e.data;
    const app = useApp.getState();
    if (m.type === 'progress') app.setEngine({ kind: 'loading', loaded: m.loaded, total: m.total });
    else if (m.type === 'ready') app.setEngine({ kind: 'ready', backend: m.backend });
    else if (m.type === 'trace') {
      app.setTrace(m.trace);
      app.setLastRunMs(m.ms);
    } else if (m.type === 'error') app.setEngine({ kind: 'error', message: m.message });
    listeners.forEach((l) => l(m));
  };
  return worker;
}

const send = (m: ToWorker) => ensureWorker().postMessage(m);

export function onEngine(fn: (m: FromWorker) => void) {
  listeners.add(fn);
  return () => void listeners.delete(fn);
}

function waitFor<T extends FromWorker['type']>(type: T | 'error'): Promise<Extract<FromWorker, { type: T }>> {
  return new Promise((resolve, reject) => {
    const off = onEngine((m) => {
      if (m.type === type) {
        off();
        resolve(m as Extract<FromWorker, { type: T }>);
      } else if (m.type === 'error') {
        off();
        reject(new Error(m.message));
      }
    });
  });
}

export function loadModel(): Promise<void> {
  if (!loading) {
    const ready = waitFor('ready');
    useApp.getState().setEngine({ kind: 'loading', loaded: 0, total: WEIGHTS_MB * 1e6 });
    send({ type: 'load', url: WEIGHTS_URL, base: import.meta.env.BASE_URL, preferGPU: !new URLSearchParams(location.search).has('cpu') });
    loading = ready.then(() => undefined).catch((e) => {
      loading = null;
      throw e;
    });
  }
  return loading;
}

export async function runPrompt(prompt: string) {
  const tok = await getTokenizer();
  const ids = tok.encode(prompt).slice(0, 64);
  if (!ids.length) return;
  await loadModel();
  const done = waitFor('trace');
  useApp.getState().setEngine({ kind: 'running' });
  send({ type: 'run', prompt, ids });
  await done;
  const s = useApp.getState();
  if (s.engine.kind === 'running') s.setEngine({ kind: 'ready', backend: s.backendName });
}

export async function generate(n: number, seed = Date.now() % 1e9) {
  await loadModel();
  const s = useApp.getState();
  const done = waitFor('trace');
  s.setEngine({ kind: 'running' });
  send({ type: 'generate', n, params: s.sampling, seed });
  await done;
  const s2 = useApp.getState();
  s2.setEngine({ kind: 'ready', backend: s2.backendName });
}
