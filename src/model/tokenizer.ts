// GPT-2 byte-level BPE, matching Hugging Face's GPT2Tokenizer exactly.
// Text → UTF-8 bytes → each byte mapped to a printable unicode char → regex pre-split →
// greedy merges by rank inside each piece → vocab ids.

export const EOT = '<|endoftext|>';
const PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/** The byte→unicode table GPT-2 uses so every byte is a visible, non-whitespace character. */
export function bytesToUnicode(): string[] {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n++);
    }
  }
  const table: string[] = new Array(256);
  bs.forEach((b, i) => (table[b] = String.fromCharCode(cs[i])));
  return table;
}

export interface Token {
  id: number;
  /** Byte-level vocab string, e.g. "ĠTower" (Ġ marks a leading space). */
  piece: string;
  /** The decoded text this token contributes, e.g. " Tower". */
  text: string;
}

export class GPT2Tokenizer {
  private encoder: Map<string, number>;
  private decoder: string[];
  private ranks: Map<string, number>;
  private byteEnc = bytesToUnicode();
  private byteDec = new Map<string, number>();
  private cache = new Map<string, string[]>();
  private utf8 = new TextEncoder();
  private utf8d = new TextDecoder('utf-8', { fatal: false });

  constructor(vocab: Record<string, number>, merges: string) {
    this.encoder = new Map(Object.entries(vocab));
    this.decoder = [];
    for (const [k, v] of this.encoder) this.decoder[v] = k;
    this.ranks = new Map();
    const lines = merges.split('\n');
    let rank = 0;
    for (const line of lines) {
      if (!line || line.startsWith('#version')) continue;
      this.ranks.set(line, rank++);
    }
    this.byteEnc.forEach((c, b) => this.byteDec.set(c, b));
  }

  get vocabSize(): number {
    return this.decoder.length;
  }

  private bpe(word: string): string[] {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + ' ' + parts[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const a = parts[best];
      const b = parts[best + 1];
      // Merge every occurrence of the pair (a, b), left to right, as HF does.
      const next: string[] = [];
      for (let i = 0; i < parts.length; ) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          next.push(a + b);
          i += 2;
        } else next.push(parts[i++]);
      }
      parts = next;
    }
    this.cache.set(word, parts);
    return parts;
  }

  encode(text: string): number[] {
    return this.tokenize(text).map((t) => t.id);
  }

  tokenize(text: string): Token[] {
    const out: Token[] = [];
    // The one special token is matched literally before BPE, as Hugging Face does.
    const parts = text.split(EOT);
    parts.forEach((part, i) => {
      if (i > 0) out.push({ id: 50256, piece: EOT, text: EOT });
      this.tokenizeOrdinary(part, out);
    });
    return out;
  }

  private tokenizeOrdinary(text: string, out: Token[]) {
    for (const m of text.matchAll(PRETOKENIZE)) {
      const mapped = Array.from(this.utf8.encode(m[0]), (b) => this.byteEnc[b]).join('');
      for (const piece of this.bpe(mapped)) {
        const id = this.encoder.get(piece);
        if (id === undefined) throw new Error(`BPE piece not in vocab: ${piece}`);
        out.push({ id, piece, text: this.decodePiece(piece) });
      }
    }
  }

  private decodePiece(piece: string): string {
    const bytes = Uint8Array.from(Array.from(piece), (c) => this.byteDec.get(c) ?? 63);
    return this.utf8d.decode(bytes);
  }

  piece(id: number): string {
    return this.decoder[id];
  }

  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) for (const c of this.decoder[id] ?? '') bytes.push(this.byteDec.get(c) ?? 63);
    return this.utf8d.decode(Uint8Array.from(bytes));
  }

  /** Decoded text of a single token (may be a partial UTF-8 sequence → replacement char). */
  tokenText(id: number): string {
    return this.decodePiece(this.decoder[id] ?? '');
  }
}
