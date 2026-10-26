import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseHeader, headerLength, assembleTensors, wantGPT2 } from '../src/model/safetensors';
import { GPT2Tokenizer } from '../src/model/tokenizer';
import type { Weights } from '../src/model/gpt2';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const WEIGHTS = resolve(ROOT, 'weights/model.safetensors');
export const hasWeights = existsSync(WEIGHTS);

let cached: Weights | null = null;
export function loadWeights(): Weights {
  if (cached) return cached;
  const buf = readFileSync(WEIGHTS);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const n = headerLength(ab.slice(0, 8));
  const h = parseHeader(ab.slice(0, 8 + n));
  const names = Object.keys(h.tensors).filter(wantGPT2);
  cached = assembleTensors(h, names, [{ start: 0, bytes: new Uint8Array(ab) }]);
  return cached;
}

export function loadTokenizer(): GPT2Tokenizer {
  const vocab = JSON.parse(readFileSync(resolve(ROOT, 'public/tokenizer/vocab.json'), 'utf8'));
  const merges = readFileSync(resolve(ROOT, 'public/tokenizer/merges.txt'), 'utf8');
  return new GPT2Tokenizer(vocab, merges);
}

export interface FixtureTensor { offset: number; shape: number[] }
export interface FixturePrompt {
  text: string; ids: number[]; tokens: string[];
  tensors: Record<string, FixtureTensor>;
  top10_last: { ids: number[]; logits: number[]; tokens: string[] };
}
export function loadFixture() {
  const index = JSON.parse(readFileSync(resolve(ROOT, 'fixtures/gpt2.json'), 'utf8')) as {
    prompts: FixturePrompt[]; tokenizer: { text: string; ids: number[] }[];
  };
  const bin = readFileSync(resolve(ROOT, 'fixtures/gpt2.bin'));
  const get = (t: FixtureTensor) => {
    const n = t.shape.reduce((a, b) => a * b, 1);
    return new Float32Array(bin.buffer.slice(bin.byteOffset + t.offset, bin.byteOffset + t.offset + n * 4));
  };
  return { ...index, get };
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>, skip?: (i: number) => boolean) {
  if (a.length !== b.length) throw new Error(`length ${a.length} vs ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    if (skip?.(i)) continue;
    m = Math.max(m, Math.abs(a[i] - b[i]));
  }
  return m;
}
