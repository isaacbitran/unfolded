// Runs GPT-2 off the main thread: streams weights (Range requests, Cache Storage), picks WebGPU or
// CPU, runs forward passes and KV-cached generation, and posts SceneTraces back.

/// <reference lib="webworker" />
import { cpuBackend, forward, mergeTrace, type Backend, type KVCache, type Trace, type Weights } from '../model/gpt2';
import { headerLength, parseHeader, planRanges, wantGPT2, type Header } from '../model/safetensors';
import { distribution, rng, sample, type SampleParams } from '../model/sample';
import { GPT2Tokenizer } from '../model/tokenizer';
import { buildSceneTrace } from '../model/trace';
import { webgpuBackend } from '../model/webgpu';
import type { FromWorker, ToWorker } from '../engine/protocol';

const post = (m: FromWorker, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

let weights: Weights | null = null;
let backend: Backend | null = null;
let tok: GPT2Tokenizer | null = null;
let last: { trace: Trace; cache: KVCache; prompt: string } | null = null;
const CACHE = 'unfolded-gpt2-v1';

async function cachedRange(url: string, start: number, end: number): Promise<Uint8Array> {
  const key = new Request(`${location.origin}/__cache__/gpt2?u=${encodeURIComponent(url)}&r=${start}-${end}`);
  const cache = 'caches' in self ? await caches.open(CACHE).catch(() => null) : null;
  const hit = await cache?.match(key);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
  if (res.status !== 206 && res.status !== 200) throw new Error(`weights request failed: HTTP ${res.status}`);
  let buf = new Uint8Array(await res.arrayBuffer());
  if (res.status === 200) buf = buf.slice(start, end); // server ignored Range
  if (buf.length !== end - start) throw new Error(`short read ${buf.length} / ${end - start}`);
  await cache?.put(key, new Response(buf)).catch(() => undefined);
  return buf;
}

async function loadTokenizer(base: string) {
  if (tok) return tok;
  const [vocab, merges] = await Promise.all([
    fetch(`${base}tokenizer/vocab.json`).then((r) => r.json()),
    fetch(`${base}tokenizer/merges.txt`).then((r) => r.text()),
  ]);
  return (tok = new GPT2Tokenizer(vocab, merges));
}

async function load(url: string, base: string, preferGPU: boolean) {
  await loadTokenizer(base);
  const first = await cachedRange(url, 0, 8);
  const n = headerLength(first.slice().buffer);
  const headBytes = await cachedRange(url, 0, 8 + n);
  const h: Header = parseHeader(headBytes.slice().buffer);
  const ranges = planRanges(h, wantGPT2);
  const total = ranges.reduce((s, r) => s + r.end - r.start, 0);
  const dataEnd = Math.max(...Object.values(h.tensors).map((t) => t.data_offsets[1])) + h.dataStart;
  // one buffer for the whole data region, so every tensor is a zero-copy Float32Array view
  const data = new Uint8Array(dataEnd - h.dataStart);
  let loaded = 0;
  post({ type: 'progress', loaded, total });
  let next = 0;
  const worker = async () => {
    while (next < ranges.length) {
      const r = ranges[next++];
      const bytes = await cachedRange(url, r.start, r.end);
      data.set(bytes, r.start - h.dataStart);
      loaded += bytes.length;
      post({ type: 'progress', loaded, total });
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]);
  const w: Weights = {};
  for (const [name, t] of Object.entries(h.tensors)) {
    if (!wantGPT2(name)) continue;
    w[name] = new Float32Array(data.buffer, t.data_offsets[0], (t.data_offsets[1] - t.data_offsets[0]) / 4);
  }
  weights = w;
  backend = (preferGPU ? await webgpuBackend(w).catch(() => null) : null) ?? cpuBackend(w);
  post({ type: 'ready', backend: backend.name });
}

async function run(prompt: string, ids: number[]) {
  if (!weights || !backend || !tok) throw new Error('model not loaded');
  const t0 = performance.now();
  const { trace, cache } = await forward(weights, backend, ids);
  last = { trace, cache, prompt };
  const scene = buildSceneTrace(prompt, trace, weights, tok);
  post({ type: 'trace', trace: scene, ms: performance.now() - t0 });
}

async function generate(n: number, params: SampleParams, seed: number) {
  if (!weights || !backend || !tok || !last) throw new Error('run a prompt first');
  const rand = rng(seed);
  let { trace, cache, prompt } = last;
  for (let i = 0; i < n; i++) {
    const V = 50257;
    const logits = trace.logits.subarray((trace.T - 1) * V, trace.T * V);
    const cands = distribution(logits, params, 20).map((c) => ({ ...c, text: tok!.tokenText(c.id) }));
    const { id, u } = sample(logits, params, rand);
    const t0 = performance.now();
    const step = await forward(weights, backend, [id], cache);
    trace = mergeTrace(trace, step.trace);
    cache = step.cache;
    prompt += tok.tokenText(id);
    post({ type: 'token', id, text: tok.tokenText(id), u, candidates: cands, ms: performance.now() - t0, cached: step.past });
  }
  last = { trace, cache, prompt };
  post({ type: 'trace', trace: buildSceneTrace(prompt, trace, weights, tok), ms: 0 });
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load(m.url, m.base, m.preferGPU);
    else if (m.type === 'run') await run(m.prompt, m.ids);
    else if (m.type === 'generate') await generate(m.n, m.params, m.seed);
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
