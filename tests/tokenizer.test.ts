import { describe, expect, it } from 'vitest';
import { loadFixture, loadTokenizer } from './helpers';
import { bytesToUnicode } from '../src/model/tokenizer';

const tok = loadTokenizer();
const fx = loadFixture();

describe('GPT-2 BPE tokenizer', () => {
  it('has the full vocabulary', () => expect(tok.vocabSize).toBe(50257));

  it('maps all 256 bytes to distinct printable characters', () => {
    const t = bytesToUnicode();
    expect(new Set(t).size).toBe(256);
    expect(t[32]).toBe('Ġ');
  });

  for (const c of fx.tokenizer) {
    it(`matches Hugging Face on ${JSON.stringify(c.text).slice(0, 40)}`, () => {
      expect(tok.encode(c.text)).toEqual(c.ids);
      expect(tok.decode(c.ids)).toBe(c.text);
    });
  }

  it('keeps the leading space on the word', () => {
    const t = tok.tokenize('The Eiffel Tower');
    expect(t.map((x) => x.text)).toEqual(['The', ' E', 'iff', 'el', ' Tower']);
  });
});
