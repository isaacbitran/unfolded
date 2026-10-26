import { describe, expect, it } from 'vitest';
import { fromHalf, toHalf, decodeSceneTrace, gridRows } from '../src/model/trace';
import { planRanges, wantGPT2, type Header } from '../src/model/safetensors';
import { readFileSync } from 'node:fs';

describe('float16 codec', () => {
  it('round-trips representative values within half precision', () => {
    for (const v of [0, 1, -1, 0.1234, -3.5, 65504, 1e-5, 123.456, -0.0009]) {
      const back = fromHalf(toHalf(v));
      expect(Math.abs(back - v)).toBeLessThanOrEqual(Math.max(Math.abs(v) * 1e-3, 6e-8));
    }
    expect(fromHalf(toHalf(Infinity))).toBe(Infinity);
  });
});

describe('precomputed demo trace', () => {
  const buf = readFileSync('public/demo/demo.trace.bin');
  const t = decodeSceneTrace(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
  it('has every tensor at the right shape', () => {
    const T = t.T;
    expect(t.tokens).toHaveLength(T);
    expect(t.resid).toHaveLength(13);
    expect(t.layers).toHaveLength(12);
    for (const l of t.layers) {
      expect(l.q.length).toBe(T * 768);
      expect(l.pattern.length).toBe(12 * T * T);
      expect(l.mlpPre.length).toBe(T * 3072);
      expect(l.mlpPost.length).toBe(T * 3072);
      expect(l.residMid.length).toBe(T * 768);
    }
    expect(t.logitsLast.length).toBe(50257);
    expect(t.positions).toHaveLength(T);
    expect(t.positions[T - 1].pNext).toBeNull();
  });
  it('attention rows still sum to 1 after float16', () => {
    const T = t.T;
    for (const l of t.layers)
      for (let h = 0; h < 12; h++)
        for (let i = 0; i < T; i++) {
          let s = 0;
          for (let j = 0; j < T; j++) s += l.pattern[(h * T + i) * T + j];
          expect(Math.abs(s - 1)).toBeLessThan(5e-3);
        }
  });
});

describe('weight download planning', () => {
  const h: Header = {
    dataStart: 100,
    tensors: {
      'wte.weight': { dtype: 'F32', shape: [2, 2], data_offsets: [0, 16] },
      'h.0.attn.bias': { dtype: 'F32', shape: [4], data_offsets: [16, 32] },
      'h.0.ln_1.weight': { dtype: 'F32', shape: [2], data_offsets: [32, 40] },
    },
  };
  it('skips the causal-mask buffers', () => {
    expect(wantGPT2('h.3.attn.bias')).toBe(false);
    expect(wantGPT2('h.3.attn.c_attn.bias')).toBe(true);
  });
  it('merges nearby tensors into one range and splits big ones', () => {
    const r = planRanges(h, wantGPT2, 64);
    expect(r).toHaveLength(1);
    expect(r[0]).toMatchObject({ start: 100, end: 140 });
    const split = planRanges(h, wantGPT2, 0, 8);
    expect(split.every((x) => x.end - x.start <= 8)).toBe(true);
  });
  it('W_E face rows avoid the prompt rows', () => {
    const rows = gridRows([13256, 8078]);
    for (const r of rows) expect(Math.min(Math.abs(r - 13256), Math.abs(r - 8078))).toBeGreaterThan(500);
  });
});
