// SceneTrace: everything the visuals need from one forward pass, plus the weight slices drawn on
// matrix faces. Built identically from a live run (worker) and for the precomputed demo (Node).

import { CFG, gelu, type LayerTrace, type Trace, type Weights } from './gpt2';
import type { GPT2Tokenizer } from './tokenizer';

export interface Preview {
  /** row and column indices shown (first 6 and last, by default) */
  rows: number[];
  cols: number[];
  /** rows.length × cols.length values */
  data: Float32Array;
  shape: [number, number];
}

export interface PositionSummary {
  /** top-10 next tokens predicted at this position (temperature 1) */
  top: { id: number; text: string; p: number }[];
  /** probability assigned to the token that actually comes next (null for the last position) */
  pNext: number | null;
}

export interface SceneTrace {
  prompt: string;
  tokens: { id: number; text: string }[];
  T: number;
  embedTok: Float32Array;
  embedPos: Float32Array;
  resid: Float32Array[];
  layers: LayerTrace[];
  lnF: Float32Array;
  lnFMean: Float32Array;
  lnFStd: Float32Array;
  logitsLast: Float32Array;
  positions: PositionSummary[];
  previews: Record<string, Preview>;
  /** γ and β of layer 0's first LayerNorm, shown in full in the LayerNorm scene */
  ln1Gamma: Float32Array;
  ln1Beta: Float32Array;
  /** decoded text of the top-200 last-position candidates, for the sampling scene */
  vocabText: Record<number, string>;
  /** true for the precomputed demo (stored as float16) */
  demo: boolean;
}

export const PREVIEW_MATRICES: Record<string, (w: Weights) => { data: Float32Array; rows: number; cols: number; colOffset?: number; colsTotal?: number }> = {
  W_E: (w) => ({ data: w['wte.weight'], rows: CFG.vocab, cols: CFG.dModel }),
  W_P: (w) => ({ data: w['wpe.weight'], rows: CFG.nCtx, cols: CFG.dModel }),
  W_Q: (w) => ({ data: w['h.0.attn.c_attn.weight'], rows: 768, cols: 768, colOffset: 0, colsTotal: 2304 }),
  W_K: (w) => ({ data: w['h.0.attn.c_attn.weight'], rows: 768, cols: 768, colOffset: 768, colsTotal: 2304 }),
  W_V: (w) => ({ data: w['h.0.attn.c_attn.weight'], rows: 768, cols: 768, colOffset: 1536, colsTotal: 2304 }),
  W_O: (w) => ({ data: w['h.0.attn.c_proj.weight'], rows: 768, cols: 768 }),
  W_up: (w) => ({ data: w['h.0.mlp.c_fc.weight'], rows: 768, cols: 3072 }),
  W_down: (w) => ({ data: w['h.0.mlp.c_proj.weight'], rows: 3072, cols: 768 }),
};

/**
 * Rows of W_E drawn faintly across the slab face: evenly spaced through the vocabulary, skipping
 * any that would sit on top of a row the prompt actually looks up.
 */
export function gridRows(ids: number[], count = 46): number[] {
  const rows: number[] = [];
  const minGap = CFG.vocab / count / 2;
  for (let i = 0; i < count; i++) {
    const r = Math.round((i * (CFG.vocab - 1)) / (count - 1));
    if (ids.some((id) => Math.abs(id - r) < minGap)) continue;
    rows.push(r);
  }
  return rows;
}

export const edgeIndices = (n: number, head = 6) => [...Array.from({ length: head }, (_, i) => i), n - 1];

export function preview(w: Weights, name: string, rowsOverride?: number[]): Preview {
  const m = PREVIEW_MATRICES[name](w);
  const rows = rowsOverride ?? edgeIndices(m.rows);
  const cols = edgeIndices(m.cols);
  const stride = m.colsTotal ?? m.cols;
  const off = m.colOffset ?? 0;
  const data = new Float32Array(rows.length * cols.length);
  rows.forEach((r, i) => cols.forEach((c, j) => (data[i * cols.length + j] = m.data[r * stride + off + c])));
  return { rows, cols, data, shape: [m.rows, m.cols] };
}

function softmaxRow(logits: Float32Array) {
  let mx = -Infinity;
  for (const l of logits) mx = Math.max(mx, l);
  const p = new Float64Array(logits.length);
  let s = 0;
  for (let i = 0; i < logits.length; i++) s += p[i] = Math.exp(logits[i] - mx);
  for (let i = 0; i < p.length; i++) p[i] /= s;
  return p;
}

function topK(p: Float64Array, k: number) {
  const idx: number[] = [];
  for (let i = 0; i < p.length; i++) {
    if (idx.length < k || p[i] > p[idx[idx.length - 1]]) {
      idx.push(i);
      idx.sort((a, b) => p[b] - p[a]);
      if (idx.length > k) idx.pop();
    }
  }
  return idx;
}

export function buildSceneTrace(prompt: string, t: Trace, w: Weights, tok: GPT2Tokenizer, demo = false): SceneTrace {
  const V = CFG.vocab;
  const positions: PositionSummary[] = [];
  const vocabText: Record<number, string> = {};
  for (let i = 0; i < t.T; i++) {
    const p = softmaxRow(t.logits.subarray(i * V, (i + 1) * V));
    const top = topK(p, i === t.T - 1 ? 200 : 10);
    const summary = top.slice(0, 10).map((id) => ({ id, text: tok.tokenText(id), p: p[id] }));
    if (i === t.T - 1) for (const id of top) vocabText[id] = tok.tokenText(id);
    positions.push({ top: summary, pNext: i < t.T - 1 ? p[t.ids[i + 1]] : null });
  }
  const previews: Record<string, Preview> = {};
  for (const name of Object.keys(PREVIEW_MATRICES)) previews[name] = preview(w, name);
  previews.W_E_grid = preview(w, 'W_E', gridRows(t.ids));
  return {
    prompt,
    tokens: t.ids.map((id) => ({ id, text: tok.tokenText(id) })),
    T: t.T,
    embedTok: t.embedTok,
    embedPos: t.embedPos,
    resid: t.resid,
    layers: t.layers,
    lnF: t.lnF,
    lnFMean: t.lnFMean,
    lnFStd: t.lnFStd,
    logitsLast: t.logits.slice((t.T - 1) * V, t.T * V),
    positions,
    previews,
    ln1Gamma: w['h.0.ln_1.weight'].slice(),
    ln1Beta: w['h.0.ln_1.bias'].slice(),
    vocabText,
    demo,
  };
}

// ---------- binary encoding (float16 payload, JSON header) ----------

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
export function toHalf(v: number): number {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e >= 31) return sign | 0x7c00 | (((x >>> 23) & 0xff) === 0xff && m ? 0x200 : 0);
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const shift = 14 - e;
    let h = m >> shift;
    if ((m >> (shift - 1)) & 1) h++;
    return sign | h;
  }
  let h = sign | (e << 10) | (m >> 13);
  if (m & 0x1000) h++; // round to nearest
  return h;
}
export function fromHalf(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// Derivable tensors are not stored: residMid = resid + attnOut, mlpPost = GELU(mlpPre).
const LAYER_KEYS: (keyof LayerTrace)[] = ['ln1', 'ln1Mean', 'ln1Std', 'q', 'k', 'v', 'scores', 'pattern', 'z', 'attnOut', 'ln2', 'ln2Mean', 'ln2Std', 'mlpPre', 'mlpOut'];

export function encodeSceneTrace(s: SceneTrace): Uint8Array {
  const arrays: Float32Array[] = [];
  const add = (a: Float32Array) => arrays.push(a) - 1;
  const header = {
    prompt: s.prompt, tokens: s.tokens, T: s.T, positions: s.positions, vocabText: s.vocabText,
    embedTok: add(s.embedTok), embedPos: add(s.embedPos), resid: s.resid.map(add),
    layers: s.layers.map((l) => Object.fromEntries(LAYER_KEYS.map((k) => [k, add(l[k])]))),
    lnF: add(s.lnF), lnFMean: add(s.lnFMean), lnFStd: add(s.lnFStd), logitsLast: add(s.logitsLast),
    ln1Gamma: add(s.ln1Gamma), ln1Beta: add(s.ln1Beta),
    previews: Object.fromEntries(Object.entries(s.previews).map(([k, p]) => [k, { rows: p.rows, cols: p.cols, shape: p.shape, data: add(p.data) }])),
    lengths: [] as number[],
  };
  header.lengths = arrays.map((a) => a.length);
  const json = new TextEncoder().encode(JSON.stringify(header));
  const total = arrays.reduce((n, a) => n + a.length, 0);
  const out = new Uint8Array(4 + json.length + ((4 - (json.length % 4)) % 4) + total * 2);
  new DataView(out.buffer).setUint32(0, json.length, true);
  out.set(json, 4);
  const start = 4 + json.length + ((4 - (json.length % 4)) % 4);
  const half = new Uint16Array(out.buffer, start, total);
  let o = 0;
  for (const a of arrays) for (let i = 0; i < a.length; i++) half[o++] = toHalf(a[i]);
  return out;
}

export function decodeSceneTrace(buf: ArrayBuffer): SceneTrace {
  const n = new DataView(buf).getUint32(0, true);
  const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  const start = 4 + n + ((4 - (n % 4)) % 4);
  const lengths: number[] = h.lengths;
  const total = lengths.reduce((a, b) => a + b, 0);
  const half = new Uint16Array(buf, start, total);
  const arrays: Float32Array[] = [];
  let o = 0;
  for (const len of lengths) {
    const a = new Float32Array(len);
    for (let i = 0; i < len; i++) a[i] = fromHalf(half[o++]);
    arrays.push(a);
  }
  const g = (i: number) => arrays[i];
  const resid: Float32Array[] = h.resid.map(g);
  const layers: LayerTrace[] = h.layers.map((l: Record<string, number>, L: number) => {
    const lt = Object.fromEntries(Object.entries(l).map(([k, i]) => [k, g(i)])) as unknown as LayerTrace;
    lt.residMid = resid[L].map((x, i) => x + lt.attnOut[i]);
    lt.mlpPost = lt.mlpPre.map(gelu);
    return lt;
  });
  return {
    prompt: h.prompt, tokens: h.tokens, T: h.T, positions: h.positions, vocabText: h.vocabText,
    embedTok: g(h.embedTok), embedPos: g(h.embedPos), resid, layers,
    lnF: g(h.lnF), lnFMean: g(h.lnFMean), lnFStd: g(h.lnFStd), logitsLast: g(h.logitsLast),
    ln1Gamma: g(h.ln1Gamma), ln1Beta: g(h.ln1Beta),
    previews: Object.fromEntries(Object.entries(h.previews as Record<string, { rows: number[]; cols: number[]; shape: [number, number]; data: number }>).map(([k, p]) => [k, { ...p, data: g(p.data) }])),
    demo: true,
  };
}
