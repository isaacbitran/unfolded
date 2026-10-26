// Precompute the demo trace with the same TypeScript engine the browser uses.
// Usage: npx tsx scripts/build-demo.ts ["prompt"]
import { writeFileSync, mkdirSync } from 'node:fs';
import { cpuBackend, forward } from '../src/model/gpt2';
import { buildSceneTrace, encodeSceneTrace, decodeSceneTrace } from '../src/model/trace';
import { loadTokenizer, loadWeights } from '../tests/helpers';

const prompt = process.argv[2] ?? 'The Eiffel Tower is in the city of';
const tok = loadTokenizer();
const w = loadWeights();
const ids = tok.encode(prompt);
const t0 = performance.now();
const { trace } = await forward(w, cpuBackend(w), ids);
const ms = performance.now() - t0;
const scene = buildSceneTrace(prompt, trace, w, tok, true);
const bin = encodeSceneTrace(scene);
mkdirSync('public/demo', { recursive: true });
writeFileSync('public/demo/demo.trace.bin', bin);
const back = decodeSceneTrace(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer);
console.log(`${JSON.stringify(prompt)} → T=${trace.T} tokens: ${scene.tokens.map((t) => JSON.stringify(t.text)).join(' ')}`);
console.log(`forward ${ms.toFixed(0)} ms (cpu); ${(bin.length / 1e6).toFixed(2)} MB`);
console.log('next:', scene.positions[trace.T - 1].top.slice(0, 5).map((c) => `${JSON.stringify(c.text)} ${(c.p * 100).toFixed(1)}%`).join(', '));
console.log('roundtrip resid12[0..3]:', Array.from(back.resid[12].slice(0, 3)), Array.from(trace.resid[12].slice(0, 3)));
