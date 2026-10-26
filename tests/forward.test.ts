import { describe, expect, it } from 'vitest';
import { hasWeights, loadFixture, loadWeights, maxAbsDiff } from './helpers';
import { CFG, cpuBackend, forward, mergeTrace, type Trace } from '../src/model/gpt2';

const fx = loadFixture();
const D = CFG.dModel;

describe.skipIf(!hasWeights)('GPT-2 forward pass vs Hugging Face', () => {
  const w = hasWeights ? loadWeights() : ({} as never);
  const be = cpuBackend(w);
  const traces = new Map<string, Trace>();
  const traceFor = async (ids: number[]) => {
    const key = ids.join(',');
    if (!traces.has(key)) traces.set(key, (await forward(w, be, ids)).trace);
    return traces.get(key)!;
  };

  for (const p of fx.prompts) {
    describe(JSON.stringify(p.text).slice(0, 30), () => {
      it('embeddings and every residual stream match', async () => {
        const t = await traceFor(p.ids);
        expect(maxAbsDiff(t.embedTok, fx.get(p.tensors.embed_tok))).toBeLessThan(1e-6);
        expect(maxAbsDiff(t.embedPos, fx.get(p.tensors.embed_pos))).toBeLessThan(1e-6);
        for (let L = 0; L <= 12; L++) {
          const ref = fx.get(p.tensors[`resid_${L}`]);
          // residual entries grow to O(100) late in the stack; compare relative to scale
          const scale = Math.max(1, ...Array.from(ref, Math.abs));
          expect(maxAbsDiff(t.resid[L], ref) / scale, `resid_${L}`).toBeLessThan(2e-5);
        }
      });

      it('every intermediate of layers 0, 5, 11 matches', async () => {
        const t = await traceFor(p.ids);
        const T = p.ids.length;
        for (const L of [0, 5, 11]) {
          const l = t.layers[L];
          const pairs: [string, Float32Array, number][] = [
            ['ln1', l.ln1, 1e-4], ['q', l.q, 1e-3], ['k', l.k, 1e-3], ['v', l.v, 1e-3],
            ['pattern', l.pattern, 1e-5], ['z', l.z, 1e-3], ['attn_out', l.attnOut, 2e-3],
            ['resid_mid', l.residMid, 5e-3], ['ln2', l.ln2, 1e-4], ['mlp_pre', l.mlpPre, 1e-3],
            ['mlp_post', l.mlpPost, 1e-3], ['mlp_out', l.mlpOut, 5e-3],
          ];
          for (const [name, mine, tol] of pairs) {
            expect(maxAbsDiff(mine, fx.get(p.tensors[`L${L}_${name}`])), `L${L} ${name}`).toBeLessThan(tol);
          }
          // raw scores: HF stores [12,T,T]; ours is identical in layout, including the upper triangle
          expect(maxAbsDiff(l.scores, fx.get(p.tensors[`L${L}_scores`])), `L${L} scores`).toBeLessThan(5e-3);
          const scaled = fx.get(p.tensors[`L${L}_scaled`]);
          expect(maxAbsDiff(l.scores.map((s) => s / 8), scaled, (i) => !Number.isFinite(scaled[i]))).toBeLessThan(1e-3);
          // every attention row sums to 1 and is zero above the diagonal
          for (let h = 0; h < 12; h++)
            for (let i = 0; i < T; i++) {
              let s = 0;
              for (let j = 0; j < T; j++) {
                const a = l.pattern[(h * T + i) * T + j];
                if (j > i) expect(a).toBe(0);
                s += a;
              }
              expect(Math.abs(s - 1)).toBeLessThan(1e-5);
            }
        }
      });

      it('logits match and the top-10 next tokens are identical', async () => {
        const t = await traceFor(p.ids);
        const ref = fx.get(p.tensors.logits);
        expect(maxAbsDiff(t.lnF, fx.get(p.tensors.ln_f))).toBeLessThan(2e-4);
        expect(maxAbsDiff(t.logits, ref)).toBeLessThan(5e-3);
        const T = p.ids.length;
        const last = t.logits.subarray((T - 1) * CFG.vocab, T * CFG.vocab);
        const order = Array.from(last.keys()).sort((a, b) => last[b] - last[a]).slice(0, 10);
        expect(order).toEqual(p.top10_last.ids);
      });
    });
  }

  it('KV-cache incremental decoding equals a full recompute', async () => {
    const ids = fx.prompts[0].ids;
    const full = await traceFor(ids);
    const first = await forward(w, be, ids.slice(0, 5));
    let merged = first.trace;
    let cache = first.cache;
    for (let i = 5; i < ids.length; i++) {
      const step = await forward(w, be, [ids[i]], cache);
      expect(step.past).toBe(i);
      merged = mergeTrace(merged, step.trace);
      cache = step.cache;
    }
    expect(merged.T).toBe(full.T);
    expect(maxAbsDiff(merged.logits, full.logits)).toBeLessThan(1e-3);
    for (const L of [0, 6, 11]) {
      expect(maxAbsDiff(merged.layers[L].pattern, full.layers[L].pattern)).toBeLessThan(1e-5);
      expect(maxAbsDiff(merged.layers[L].scores, full.layers[L].scores)).toBeLessThan(1e-3);
      expect(maxAbsDiff(merged.layers[L].mlpPost, full.layers[L].mlpPost)).toBeLessThan(1e-3);
    }
    expect(cache.k[0].length).toBe(ids.length * D);
  });
});
