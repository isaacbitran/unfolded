// Turning logits into a next token: temperature, top-k, top-p (nucleus), then a draw.

export interface SampleParams {
  temperature: number; // 0 → greedy
  topK: number; // 0 → off
  topP: number; // 1 → off
}

export interface Candidate {
  id: number;
  logit: number;
  /** softmax(logits / T) over the whole vocabulary, before any filtering */
  prob: number;
  /** probability after top-k/top-p filtering and renormalisation (0 if filtered out) */
  kept: number;
  /** cumulative probability (sorted, unfiltered) up to and including this token */
  cum: number;
}

/** Ranked candidates (descending) with every stage of the sampling pipeline exposed. */
export function distribution(logits: Float32Array, p: SampleParams, limit = 50): Candidate[] {
  const n = logits.length;
  const tau = Math.max(p.temperature, 1e-6);
  let mx = -Infinity;
  for (let i = 0; i < n; i++) mx = Math.max(mx, logits[i]);
  const probs = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp((logits[i] - mx) / tau);
    probs[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) probs[i] /= sum;

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
  const out: Candidate[] = [];
  let cum = 0;
  let keptMass = 0;
  const keep: boolean[] = [];
  for (let r = 0; r < n; r++) {
    const id = order[r];
    const before = cum;
    cum += probs[id];
    const inK = !p.topK || r < p.topK;
    const inP = p.topP >= 1 || before < p.topP; // keep the token that crosses the threshold
    const k = inK && inP && (p.temperature > 0 || r === 0);
    keep.push(k);
    if (k) keptMass += probs[id];
    if (r < limit) out.push({ id, logit: logits[id], prob: probs[id], kept: 0, cum });
  }
  for (let r = 0; r < out.length; r++) out[r].kept = keep[r] ? out[r].prob / keptMass : 0;
  return out;
}

export function topIndices(values: ArrayLike<number>, k: number): number[] {
  const n = values.length;
  k = Math.min(k, n);
  // Threshold via a quickselect-free approach: collect candidates above a rising floor.
  const idx: number[] = [];
  let floor = -Infinity;
  for (let i = 0; i < n; i++) {
    if (values[i] <= floor && idx.length >= k) continue;
    idx.push(i);
    if (idx.length >= 4 * k) {
      idx.sort((a, b) => values[b] - values[a]);
      idx.length = k;
      floor = values[idx[k - 1]];
    }
  }
  idx.sort((a, b) => values[b] - values[a]);
  idx.length = Math.min(k, idx.length);
  return idx;
}

/** mulberry32 — small seeded RNG so a demo replays identically. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Draw one token. Returns the id and the uniform number used (shown in the sampling scene). */
export function sample(logits: Float32Array, p: SampleParams, rand: () => number) {
  const cands = distribution(logits, p, logits.length);
  if (p.temperature <= 0) return { id: cands[0].id, u: 0 };
  const u = rand();
  let acc = 0;
  for (const c of cands) {
    acc += c.kept;
    if (u < acc) return { id: c.id, u };
  }
  const last = cands.filter((c) => c.kept > 0).pop()!;
  return { id: last.id, u };
}
