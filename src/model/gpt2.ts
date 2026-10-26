// GPT-2 small forward pass that keeps every intermediate tensor.
// Layout convention: activations are row-major [T, D] Float32Arrays; attention tensors are [H, T, T].

export const CFG = {
  nLayer: 12,
  nHead: 12,
  dModel: 768,
  dHead: 64,
  dMlp: 3072,
  vocab: 50257,
  nCtx: 1024,
  eps: 1e-5,
} as const;

export type Weights = Record<string, Float32Array>;

export interface LayerTrace {
  ln1: Float32Array; // [T,768]
  ln1Mean: Float32Array; // [T]
  ln1Std: Float32Array; // [T]  sqrt(var + eps)
  q: Float32Array; // [T,768]
  k: Float32Array;
  v: Float32Array;
  scores: Float32Array; // [H,T,T] raw q·k (upper triangle filled too; it is masked afterwards)
  pattern: Float32Array; // [H,T,T] softmax(scores/8 + mask)
  z: Float32Array; // [T,768] concat of per-head weighted value sums
  attnOut: Float32Array; // [T,768] z·W_O + b_O
  residMid: Float32Array; // [T,768]
  ln2: Float32Array;
  ln2Mean: Float32Array;
  ln2Std: Float32Array;
  mlpPre: Float32Array; // [T,3072]
  mlpPost: Float32Array; // [T,3072] GELU
  mlpOut: Float32Array; // [T,768]
}

export interface Trace {
  T: number;
  ids: number[];
  embedTok: Float32Array; // [T,768]
  embedPos: Float32Array; // [T,768]
  /** resid[0] = embeddings, resid[L+1] = stream after block L. */
  resid: Float32Array[];
  layers: LayerTrace[];
  lnF: Float32Array; // [T,768]
  lnFMean: Float32Array;
  lnFStd: Float32Array;
  logits: Float32Array; // [T,50257]
}

/** Matmul backend. `x` is [T,inDim]; W is stored [inDim,outDim] (GPT-2 Conv1D layout). */
export interface Backend {
  name: string;
  linear(x: Float32Array, T: number, weight: string, bias: string, inDim: number, outDim: number): Promise<Float32Array>;
  /** logits[t,v] = Σ_i x[t,i]·wte[v,i] */
  unembed(x: Float32Array, T: number): Promise<Float32Array>;
}

export function linearCPU(x: Float32Array, T: number, W: Float32Array, b: Float32Array, inDim: number, outDim: number): Float32Array {
  const y = new Float32Array(T * outDim);
  for (let t = 0; t < T; t++) y.set(b, t * outDim);
  // i-outer keeps one W row (≤12 KB) hot in cache while it is applied to every token.
  for (let i = 0; i < inDim; i++) {
    const wOff = i * outDim;
    for (let t = 0; t < T; t++) {
      const xi = x[t * inDim + i];
      if (xi === 0) continue;
      const yOff = t * outDim;
      for (let o = 0; o < outDim; o++) y[yOff + o] += xi * W[wOff + o];
    }
  }
  return y;
}

export function unembedCPU(x: Float32Array, T: number, wte: Float32Array): Float32Array {
  const { vocab, dModel: D } = CFG;
  const out = new Float32Array(T * vocab);
  for (let v = 0; v < vocab; v++) {
    const off = v * D;
    for (let t = 0; t < T; t++) {
      const xo = t * D;
      let s = 0;
      for (let i = 0; i < D; i++) s += x[xo + i] * wte[off + i];
      out[t * vocab + v] = s;
    }
  }
  return out;
}

export function cpuBackend(w: Weights): Backend {
  return {
    name: 'cpu',
    async linear(x, T, weight, bias, inDim, outDim) {
      return linearCPU(x, T, w[weight], w[bias], inDim, outDim);
    },
    async unembed(x, T) {
      return unembedCPU(x, T, w['wte.weight']);
    },
  };
}

export function layerNorm(x: Float32Array, T: number, g: Float32Array, b: Float32Array, D = CFG.dModel) {
  const y = new Float32Array(T * D);
  const mean = new Float32Array(T);
  const std = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let m = 0;
    for (let i = 0; i < D; i++) m += x[o + i];
    m /= D;
    let v = 0;
    for (let i = 0; i < D; i++) {
      const d = x[o + i] - m;
      v += d * d;
    }
    v /= D;
    const s = Math.sqrt(v + CFG.eps);
    mean[t] = m;
    std[t] = s;
    for (let i = 0; i < D; i++) y[o + i] = ((x[o + i] - m) / s) * g[i] + b[i];
  }
  return { y, mean, std };
}

const GELU_C = Math.sqrt(2 / Math.PI);
export const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));

/** Per-layer keys and values of every position processed so far: [T,768] each. */
export interface KVCache {
  T: number;
  k: Float32Array[];
  v: Float32Array[];
}

export const emptyCache = (): KVCache => ({ T: 0, k: [], v: [], });

function concatRows(a: Float32Array | undefined, b: Float32Array): Float32Array {
  if (!a || a.length === 0) return b;
  const out = new Float32Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}

/**
 * Run positions [cache.T, cache.T + ids.length) through the model.
 * Returns a trace covering only the new positions (attention tensors are [H, Tnew, Tall]) and the
 * updated cache. With an empty cache this is the ordinary full forward pass.
 */
export async function forward(w: Weights, be: Backend, ids: number[], cache: KVCache = emptyCache()) {
  const { dModel: D, nHead: H, dHead: Dh, dMlp: M, nLayer } = CFG;
  const T = ids.length;
  const P = cache.T; // past length
  const S = P + T; // total length
  if (S > CFG.nCtx) throw new Error(`context ${S} exceeds ${CFG.nCtx}`);

  const wte = w['wte.weight'];
  const wpe = w['wpe.weight'];
  const embedTok = new Float32Array(T * D);
  const embedPos = new Float32Array(T * D);
  const x0 = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    embedTok.set(wte.subarray(ids[t] * D, ids[t] * D + D), t * D);
    embedPos.set(wpe.subarray((P + t) * D, (P + t) * D + D), t * D);
    for (let i = 0; i < D; i++) x0[t * D + i] = embedTok[t * D + i] + embedPos[t * D + i];
  }

  const resid: Float32Array[] = [x0];
  const layers: LayerTrace[] = [];
  const next: KVCache = { T: S, k: [], v: [] };
  let x = x0;
  const scale = 1 / Math.sqrt(Dh);

  for (let L = 0; L < nLayer; L++) {
    const p = `h.${L}.`;
    const n1 = layerNorm(x, T, w[p + 'ln_1.weight'], w[p + 'ln_1.bias']);
    const qkv = await be.linear(n1.y, T, p + 'attn.c_attn.weight', p + 'attn.c_attn.bias', D, 3 * D);
    const q = new Float32Array(T * D);
    const k = new Float32Array(T * D);
    const v = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      q.set(qkv.subarray(t * 3 * D, t * 3 * D + D), t * D);
      k.set(qkv.subarray(t * 3 * D + D, t * 3 * D + 2 * D), t * D);
      v.set(qkv.subarray(t * 3 * D + 2 * D, t * 3 * D + 3 * D), t * D);
    }
    const kAll = concatRows(cache.k[L], k);
    const vAll = concatRows(cache.v[L], v);
    next.k[L] = kAll;
    next.v[L] = vAll;

    const scores = new Float32Array(H * T * S);
    const pattern = new Float32Array(H * T * S);
    const z = new Float32Array(T * D);
    for (let h = 0; h < H; h++) {
      const ho = h * Dh;
      for (let t = 0; t < T; t++) {
        const row = (h * T + t) * S;
        const qo = t * D + ho;
        for (let j = 0; j < S; j++) {
          const ko = j * D + ho;
          let s = 0;
          for (let d = 0; d < Dh; d++) s += q[qo + d] * kAll[ko + d];
          scores[row + j] = s;
        }
        // causal softmax over j ≤ P + t
        const last = P + t;
        let mx = -Infinity;
        for (let j = 0; j <= last; j++) mx = Math.max(mx, scores[row + j] * scale);
        let sum = 0;
        for (let j = 0; j <= last; j++) {
          const e = Math.exp(scores[row + j] * scale - mx);
          pattern[row + j] = e;
          sum += e;
        }
        for (let j = 0; j <= last; j++) pattern[row + j] /= sum;
        const zo = t * D + ho;
        for (let j = 0; j <= last; j++) {
          const a = pattern[row + j];
          const vo = j * D + ho;
          for (let d = 0; d < Dh; d++) z[zo + d] += a * vAll[vo + d];
        }
      }
    }
    const attnOut = await be.linear(z, T, p + 'attn.c_proj.weight', p + 'attn.c_proj.bias', D, D);
    const residMid = new Float32Array(T * D);
    for (let i = 0; i < T * D; i++) residMid[i] = x[i] + attnOut[i];

    const n2 = layerNorm(residMid, T, w[p + 'ln_2.weight'], w[p + 'ln_2.bias']);
    const mlpPre = await be.linear(n2.y, T, p + 'mlp.c_fc.weight', p + 'mlp.c_fc.bias', D, M);
    const mlpPost = new Float32Array(T * M);
    for (let i = 0; i < T * M; i++) mlpPost[i] = gelu(mlpPre[i]);
    const mlpOut = await be.linear(mlpPost, T, p + 'mlp.c_proj.weight', p + 'mlp.c_proj.bias', M, D);
    const xNext = new Float32Array(T * D);
    for (let i = 0; i < T * D; i++) xNext[i] = residMid[i] + mlpOut[i];

    layers.push({
      ln1: n1.y, ln1Mean: n1.mean, ln1Std: n1.std, q, k, v, scores, pattern, z, attnOut, residMid,
      ln2: n2.y, ln2Mean: n2.mean, ln2Std: n2.std, mlpPre, mlpPost, mlpOut,
    });
    resid.push(xNext);
    x = xNext;
  }

  const nf = layerNorm(x, T, w['ln_f.weight'], w['ln_f.bias']);
  const logits = await be.unembed(nf.y, T);
  const trace: Trace = {
    T, ids: ids.slice(), embedTok, embedPos, resid, layers,
    lnF: nf.y, lnFMean: nf.mean, lnFStd: nf.std, logits,
  };
  return { trace, cache: next, past: P };
}

/** Full-prompt trace (attention tensors [H,T,T]). */
export async function run(w: Weights, be: Backend, ids: number[]) {
  return forward(w, be, ids);
}

function appendRows(a: Float32Array, b: Float32Array) {
  return concatRows(a, b);
}

/**
 * Merge a trace of new positions (computed with a KV cache) into the full trace of the earlier
 * positions, producing exactly what a full recompute would. Earlier rows are untouched — the
 * causal mask guarantees they cannot depend on later tokens. The only new numbers for old rows are
 * the raw (masked-away) scores q_old·k_new in the upper triangle, which we fill in for display.
 */
export function mergeTrace(old: Trace, inc: Trace): Trace {
  const { nHead: H, dHead: Dh, dModel: D } = CFG;
  const P = old.T;
  const T = inc.T;
  const S = P + T;
  const layers = old.layers.map((lo, L) => {
    const li = inc.layers[L];
    const scores = new Float32Array(H * S * S);
    const pattern = new Float32Array(H * S * S);
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < P; i++) {
        const dst = (h * S + i) * S;
        scores.set(lo.scores.subarray((h * P + i) * P, (h * P + i) * P + P), dst);
        pattern.set(lo.pattern.subarray((h * P + i) * P, (h * P + i) * P + P), dst);
        for (let j = P; j < S; j++) {
          let s = 0;
          for (let d = 0; d < Dh; d++) s += lo.q[i * D + h * Dh + d] * li.k[(j - P) * D + h * Dh + d];
          scores[dst + j] = s;
        }
      }
      for (let t = 0; t < T; t++) {
        const dst = (h * S + P + t) * S;
        scores.set(li.scores.subarray((h * T + t) * S, (h * T + t) * S + S), dst);
        pattern.set(li.pattern.subarray((h * T + t) * S, (h * T + t) * S + S), dst);
      }
    }
    const merged = {} as LayerTrace;
    for (const key of Object.keys(lo) as (keyof LayerTrace)[]) merged[key] = appendRows(lo[key], li[key]);
    merged.scores = scores;
    merged.pattern = pattern;
    return merged;
  });
  return {
    T: S,
    ids: [...old.ids, ...inc.ids],
    embedTok: appendRows(old.embedTok, inc.embedTok),
    embedPos: appendRows(old.embedPos, inc.embedPos),
    resid: old.resid.map((r, i) => appendRows(r, inc.resid[i])),
    layers,
    lnF: appendRows(old.lnF, inc.lnF),
    lnFMean: appendRows(old.lnFMean, inc.lnFMean),
    lnFStd: appendRows(old.lnFStd, inc.lnFStd),
    logits: appendRows(old.logits, inc.logits),
  };
}
