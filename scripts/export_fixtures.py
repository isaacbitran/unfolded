"""Export reference GPT-2 activations from Hugging Face transformers for the TypeScript tests.

Writes fixtures/gpt2.json (index + token data) and fixtures/gpt2.bin (float32 tensors).
Every intermediate is computed step by step from the HF model's own modules and the
final logits are checked against model(input_ids).logits, so the fixture is anchored to HF.

Run: ~/ClaudeCode/emLocal/.venv/bin/python scripts/export_fixtures.py
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures"
PROMPTS = [
    "The quick brown fox jumps over the lazy",
    "Harry Potter and the",
    " héllo,  wörld!! 12345 can't won't\n\tdon't",
]
# Tokenizer-only cases (no activations), to exercise byte-level BPE edge cases.
TOKENIZER_CASES = PROMPTS + [
    "",
    " ",
    "Hello world",
    "   leading spaces and trailing   ",
    "emoji 🤖🧠 and CJK 漢字 and Ελληνικά",
    "I'm you're they'll we'd it's",
    "x = y**2 + 3.14159; // comment",
    "newlines\n\n\nand\r\ntabs\t\t",
    "<|endoftext|> special?",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
]
FULL_LAYERS = {0, 5, 11}  # every intermediate is stored for these layers; residuals for all


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", torch_dtype=torch.float32).eval()
    tr = model.transformer

    blob = bytearray()
    index: dict = {"prompts": [], "tokenizer": []}

    def put(arr: torch.Tensor) -> dict:
        a = arr.detach().to(torch.float32).contiguous().numpy()
        entry = {"offset": len(blob), "shape": list(a.shape)}
        blob.extend(a.astype("<f4").tobytes())
        return entry

    for text in TOKENIZER_CASES:
        ids = tok.encode(text)
        index["tokenizer"].append({"text": text, "ids": ids, "tokens": tok.convert_ids_to_tokens(ids)})

    with torch.no_grad():
        for text in PROMPTS:
            ids = tok.encode(text)
            x_ids = torch.tensor([ids])
            T = len(ids)
            t: dict = {}
            emb_tok = tr.wte(x_ids)[0]
            emb_pos = tr.wpe.weight[:T]
            t["embed_tok"] = put(emb_tok)
            t["embed_pos"] = put(emb_pos)
            x = emb_tok + emb_pos
            t["resid_0"] = put(x)
            for L, blk in enumerate(tr.h):
                full = L in FULL_LAYERS
                ln1 = blk.ln_1(x)
                qkv = blk.attn.c_attn(ln1)
                q, k, v = qkv.split(768, dim=-1)
                qh = q.view(T, 12, 64).transpose(0, 1)
                kh = k.view(T, 12, 64).transpose(0, 1)
                vh = v.view(T, 12, 64).transpose(0, 1)
                scores = qh @ kh.transpose(-1, -2)  # raw dot products
                scaled = scores / math.sqrt(64)
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
                masked = scaled.masked_fill(mask, float("-inf"))
                pattern = torch.softmax(masked, dim=-1)
                z = pattern @ vh  # [12, T, 64]
                z_cat = z.transpose(0, 1).reshape(T, 768)
                attn_out = blk.attn.c_proj(z_cat)
                x_mid = x + attn_out
                ln2 = blk.ln_2(x_mid)
                pre = blk.mlp.c_fc(ln2)
                post = blk.mlp.act(pre)
                mlp_out = blk.mlp.c_proj(post)
                x_new = x_mid + mlp_out
                if full:
                    p = f"L{L}_"
                    t[p + "ln1"] = put(ln1)
                    t[p + "q"] = put(q)
                    t[p + "k"] = put(k)
                    t[p + "v"] = put(v)
                    t[p + "scores"] = put(scores)
                    t[p + "scaled"] = put(scaled)
                    t[p + "pattern"] = put(pattern)
                    t[p + "z"] = put(z_cat)
                    t[p + "attn_out"] = put(attn_out)
                    t[p + "resid_mid"] = put(x_mid)
                    t[p + "ln2"] = put(ln2)
                    t[p + "mlp_pre"] = put(pre)
                    t[p + "mlp_post"] = put(post)
                    t[p + "mlp_out"] = put(mlp_out)
                x = x_new
                t[f"resid_{L + 1}"] = put(x)
            lnf = tr.ln_f(x)
            t["ln_f"] = put(lnf)
            logits = lnf @ tr.wte.weight.T
            ref = model(x_ids).logits[0]
            err = (logits - ref).abs().max().item()
            assert err < 1e-3, f"step-by-step logits drift from HF by {err}"
            t["logits"] = put(logits)
            top = torch.topk(logits[-1], 10)
            index["prompts"].append({
                "text": text,
                "ids": ids,
                "tokens": tok.convert_ids_to_tokens(ids),
                "tensors": t,
                "top10_last": {"ids": top.indices.tolist(), "logits": top.values.tolist(),
                                "tokens": [tok.decode([i]) for i in top.indices.tolist()]},
                "hf_max_abs_err": err,
            })
            print(f"{text!r}: T={T} top1={tok.decode([top.indices[0].item()])!r} hf_err={err:.2e}")

    (OUT / "gpt2.bin").write_bytes(bytes(blob))
    (OUT / "gpt2.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(f"wrote {len(blob) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
