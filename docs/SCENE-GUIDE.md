# Writing a walkthrough scene

Read `DESIGN.md` §3 (visual language) and §5 (the storyboard entry for your step) first. Then read
`src/scene/steps/S02Embedding.tsx` (the reference scene) and `S03Position.tsx` (uses the shared
primitives).

## Contract

- One file per step: `src/scene/steps/SNNName.tsx`, exporting `function SNNName({ index, trace })`.
  It is already registered in `src/scene/World.tsx` with its index.
- Everything animates as a **pure function of progress** `p ∈ [0,1]` inside `useStep(index, (p, vis) => …)`.
  `vis` is the crossfade opacity (0 when the story is elsewhere): multiply every opacity by it.
  Never use timers, `Date.now`, or React state for animation. Scrolling backwards must play it in reverse.
- Mutate imperatively in the `useStep` callback via handles (`setOpacity`, `setProgress`, `fade(label, a)`,
  group positions). Do not re-render React per frame.
- Use `seg(p, a, b)` for a 0→1 window, `easeInOut`, `env`. Windows should overlap so motion is continuous.
- Real numbers only: read tensors from `trace` (`SceneTrace` in `src/model/trace.ts`). Layer 0 is
  "block 1" in the prose. Attention tensors are `[12, T, T]` row-major; activations `[T, 768]`.
  Weight corners for slab faces are in `trace.previews` (`W_Q, W_K, W_V, W_O, W_up, W_down, W_E, W_P`).
- Your step's camera keys live in `src/story/steps.ts` (`cam:` of your step). Edit **only your
  step's `cam` block and, if needed, its prose**. Keys use story-space progress `p`; the camera eases
  between keys and stops exactly on each key. Your first key should be close to the previous step's
  last key so the cut is a glide.
- Coordinates: token columns at `x = colX(k, T)` (`theme.ts`), first entry at `y = COL_TOP` (1.25),
  rows `L.rowH` apart, token labels at `TOKEN_LABEL_Y`. The text column covers the left ~34% of the
  screen; the camera already applies a lens shift for it, so frame your scene around its own center.
- Colors: only from `C` in `theme.ts` (query blue, key red, value green, attention/W_O purple, MLP gold,
  LayerNorm silver, residual warm white, position amber). Brightness carries meaning: active glows.

## Primitives (src/scene/primitives, src/scene/glyphs)

- `Columns` — bracketed column vectors from a `[T, D]` tensor (first 6 entries, ⋮, last), optional
  flight from other positions (`from`, `delay`) driven by `setProgress`. Use `offset`/`len` for a head slice.
- `MatrixSlab` — a weight matrix as a glass slab with its real corner entries, name and true shape.
- `Glass` — bare glass box (`setOpacity`, `setBand`). `Lines` + `bracketPoints` for hairlines/brackets.
- `GlyphField` + `layoutText` — any numbers/text in the atlas charset, instanced, with per-glyph
  `from`/`delay`/`appear`/`vanish` windows. One field can hold thousands of glyphs: prefer few fields.
- `Label` (+ `fade`) — DOM KaTeX/text pinned to a 3D point. Use for math, shapes, captions.
- `TextPlane` — arbitrary unicode text as a 3D plane (token strings).
- `TokenHeader` — token labels above the columns, ids, hairlines.

## Checking your work

The dev server runs at http://localhost:5391 (do not start another on that port). Render frames:

```
node scripts/shots.mjs 7.1 7.4 7.7 7.97 --url http://localhost:5391     # → shots/s7.1.png …
~/ClaudeCode/edgeHunt/.venv/bin/python scripts/sheet.py shots/sheet-7.png shots/s7.1.png …
```

Look at every frame. Check: nothing hidden under the text column, nothing clipped, labels legible,
numbers not overlapping at rest, the first and last frames match the neighbouring steps, and it looks
calm and deliberate rather than busy. `npx tsc -p tsconfig.app.json --noEmit` must pass.
