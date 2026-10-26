// Minimal safetensors reader: parse the JSON header, plan Range requests for the tensors we need,
// and expose Float32Array views.

export interface TensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface Header {
  /** Byte offset where tensor data starts (8 + header length). */
  dataStart: number;
  tensors: Record<string, TensorInfo>;
}

export function parseHeader(prefix: ArrayBuffer): Header {
  const view = new DataView(prefix);
  const n = Number(view.getBigUint64(0, true));
  if (prefix.byteLength < 8 + n) throw new Error(`need ${8 + n} header bytes, got ${prefix.byteLength}`);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(prefix, 8, n)));
  delete json.__metadata__;
  return { dataStart: 8 + n, tensors: json };
}

/** Header length from the first 8 bytes. */
export function headerLength(first8: ArrayBuffer): number {
  return Number(new DataView(first8).getBigUint64(0, true));
}

export interface ByteRange {
  start: number; // absolute file offset, inclusive
  end: number; // exclusive
  names: string[];
}

/**
 * Group the wanted tensors into contiguous byte ranges (merging gaps smaller than `maxGap`),
 * then split ranges larger than `maxChunk` so progress is smooth and requests stay cacheable.
 */
export function planRanges(h: Header, want: (name: string) => boolean, maxGap = 1 << 16, maxChunk = 32 << 20): ByteRange[] {
  const items = Object.entries(h.tensors)
    .filter(([name]) => want(name))
    .map(([name, t]) => ({ name, start: h.dataStart + t.data_offsets[0], end: h.dataStart + t.data_offsets[1] }))
    .sort((a, b) => a.start - b.start);
  const merged: ByteRange[] = [];
  for (const it of items) {
    const last = merged[merged.length - 1];
    if (last && it.start - last.end <= maxGap) {
      last.end = Math.max(last.end, it.end);
      last.names.push(it.name);
    } else merged.push({ start: it.start, end: it.end, names: [it.name] });
  }
  const out: ByteRange[] = [];
  for (const r of merged) {
    for (let s = r.start; s < r.end; s += maxChunk) out.push({ start: s, end: Math.min(r.end, s + maxChunk), names: r.names });
  }
  return out;
}

/** Copy each wanted tensor's bytes out of the downloaded chunks into its own Float32Array. */
export function assembleTensors(
  h: Header,
  names: string[],
  chunks: { start: number; bytes: Uint8Array }[],
): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {};
  const sorted = chunks.slice().sort((a, b) => a.start - b.start);
  for (const name of names) {
    const t = h.tensors[name];
    if (t.dtype !== 'F32') throw new Error(`${name}: unsupported dtype ${t.dtype}`);
    const s = h.dataStart + t.data_offsets[0];
    const e = h.dataStart + t.data_offsets[1];
    const dst = new Float32Array((e - s) / 4);
    const dstBytes = new Uint8Array(dst.buffer);
    for (const c of sorted) {
      const cs = c.start;
      const ce = c.start + c.bytes.length;
      const a = Math.max(s, cs);
      const b = Math.min(e, ce);
      if (a < b) dstBytes.set(c.bytes.subarray(a - cs, b - cs), a - s);
    }
    out[name] = dst;
  }
  return out;
}

/** GPT-2 needs every tensor except the per-layer causal-mask buffers (`h.N.attn.bias`, 4 MB each). */
export const wantGPT2 = (name: string) => !/^h\.\d+\.attn\.bias$/.test(name) && !name.endsWith('masked_bias');
