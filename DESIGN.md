# Unfolded — design

GPT-2 small, running in the browser, filmed one operation at a time.

This document is the plan: the storyboard, the visual language, the app architecture, and the data
flow from the forward pass to the pixels. Code follows it; when they disagree, fix one of them.

---

## 1. What we are making and for whom

A smart beginner types a sentence and watches GPT-2 small (124M parameters) turn it into a
probability for every one of 50,257 possible next tokens. Every number on screen is a real
activation from that forward pass. Every tensor is labeled with its true shape. Nothing is
schematic unless it says so.

Two modes share one 3D world:

- **Walkthrough**: 17 scroll-driven steps (arrow keys also work). The camera flies through the model.
  A text column explains each step in plain sentences plus KaTeX math.
- **Explore**: type any prompt, orbit the whole 12-layer tower, click a layer, a head, or a token to
  fly to it, scrub the residual stream through the layers, and generate tokens with temperature,
  top-k and top-p.

What existing explainers do well and where they confuse people (from studying them):

| Explainer | Great at | Where learners get lost |
|---|---|---|
| 3D cinematic videos (glass slabs, columns of numbers) | Making matrices feel physical; showing scale ("50,257"); calm pacing | Not interactive; numbers are illustrative, not from your prompt |
| Transformer Explainer (2D, GPT-2 in browser) | Real model, your prompt, sampling controls | Q/K/V shown as flows without the matrix multiply; one layer only; shapes mostly hidden |
| bbycroft LLM viz (3D, nano-GPT) | Every tensor present, true scale | Overwhelming; little narrative; tiny model |
| Illustrated Transformer (static) | Clear step order | Encoder–decoder, not GPT; no real numbers |

Our gap: real GPT-2 numbers **and** cinematic 3D **and** every shape shown **and** a narrative that
answers the question a learner actually has at each step.

---

## 2. GPT-2 small, precisely

Configuration: `n_layer=12, n_head=12, d_model=768, d_head=64, d_mlp=3072, vocab=50257, n_ctx=1024`,
LayerNorm ε = 1e-5, GELU = tanh approximation (`gelu_new`), pre-LN blocks, tied unembedding.

For a prompt of T tokens with ids `t_1..t_T`:

```
x⁰ = W_E[t] + W_P[0..T)                              [T, 768]
for L in 0..11:
  a  = LN₁(xᴸ)                                       [T, 768]
  Q,K,V = a·W_Q + b_Q, a·W_K + b_K, a·W_V + b_V      [T, 768] each; c_attn packs them as [768, 2304]
  split into 12 heads of 64:                         [12, T, 64]
  S_h = Q_h K_hᵀ                                     [12, T, T]   raw dot products
  S_h / √64 → causal mask (j > i ⇒ −∞) → softmax     [12, T, T]   rows sum to 1
  Z_h = A_h V_h                                      [12, T, 64]
  concat → Z·W_O + b_O                               [T, 768]
  xᴸ' = xᴸ + attn_out                                residual add
  m  = LN₂(xᴸ')
  h  = GELU(m·W_up + b_up)                           [T, 3072]
  xᴸ⁺¹ = xᴸ' + h·W_down + b_down                     [T, 768]
y = LN_f(x¹²)
logits = y · W_Eᵀ                                    [T, 50257]  (W_U = W_Eᵀ, tied)
p = softmax(logits_T / temperature) after top-k / top-p filtering
```

Safetensors names (`openai-community/gpt2`): `wte.weight [50257,768]`, `wpe.weight [1024,768]`,
`h.{L}.ln_1.{weight,bias}`, `h.{L}.attn.c_attn.{weight [768,2304], bias}`,
`h.{L}.attn.c_proj.{weight [768,768], bias}`, `h.{L}.ln_2.*`, `h.{L}.mlp.c_fc.{weight [768,3072], bias}`,
`h.{L}.mlp.c_proj.{weight [3072,768], bias}`, `ln_f.*`. Conv1D weights are stored `[in, out]`, so
`y = x·W + b` with no transpose. `h.{L}.attn.bias` is a 4 MB causal-mask buffer per layer that we
never download (48 MB saved via HTTP Range requests).

---

## 3. Visual language

**Stage.** Pure black (`#000`). No floor, no grid. Depth comes from fog-free darkness, translucent
glass, and a slow camera. Bloom only catches bright things (active numbers, edges), threshold high.

**Glass slabs** (matrices, blocks). A custom shader, not physical transmission (too slow at 60 fps
for dozens of slabs): faint fill (α≈0.06), Fresnel rim brightening at grazing angles, crisp 1 px
edge lines, tinted by role. Slabs have thickness proportional to nothing; their face size is
proportional to the matrix shape on a log-ish scale so 50,257 × 768 reads as *tall*.

**Numbers.** The signature element. A serif face (KaTeX Main, a Computer Modern clone) rendered
through one shared glyph atlas into instanced quads: one draw call per vector or matrix, tens of
thousands of glyphs at 60 fps. Brightness encodes |value| relative to the vector's scale, so the
large entries glow. Negative numbers use a true minus sign (−), not a hyphen. Format: 1 decimal
for |x| ≥ 10, else 2 decimals, `+0.42 / −1.07` style without the plus.

**Vectors.** A column of 8 visible entries — the first 6, a vertical ellipsis ⋮, the last — inside
tall square brackets, exactly as in the reference frames. The true shape sits under it in small
italic math: `768`. Token label floats above with a hairline dropping into its column.

**Typography.** All in-scene math is KaTeX rendered to the DOM and pinned to 3D anchors (crisp at
any DPI). Labels: white, 85% opacity. Shapes: 55% opacity, italic. The prose column uses a serif
body (Source Serif 4) and KaTeX for inline math.

**Color code** (used for the glyph tint, slab tint, and flow particles; nothing else is colored):

| Role | Color | Hex |
|---|---|---|
| Token / residual stream | warm white | `#F4EEE2` |
| Positional embedding | amber | `#F2B45C` |
| LayerNorm | silver | `#B9C3CF` |
| Query | blue | `#5DA9FF` |
| Key | red | `#FF6B6B` |
| Value | green | `#5FD68A` |
| Attention output / W_O | purple | `#B48CFF` |
| MLP | orange-gold | `#FFB454` |
| Logits / probabilities | white → highlight `#FFFFFF` | |

**Motion.** Every transition is a pure function of progress `p ∈ [0,1]` within a step, so scrolling
backwards plays it in reverse and nothing pops. Numbers *fly*: a glyph instance has a `from` and
`to` position and a per-instance stagger; the GPU interpolates. Easing: `easeInOutCubic` for
moves, `easeOutQuart` for appear. No cuts — steps overlap by 15% so the next step's objects fade in
while the previous ones finish.

**Restraint rules.** At most ~3 labeled objects in focus at once. Anything out of focus dims to 15%.
No legends, no arrows with text on them; the prose explains, the scene shows.

---

## 4. The world (one coordinate system)

```
            y ↑
              |      token columns spaced 1.7 along +x (token i at x = (i − (T−1)/2)·1.7)
              |
   ──────────[ x⁰ ]────[ block 1 ]────[ block 2 ]── … ──[ block 12 ]────[ LN_f · W_U ]──→ −z
   z = 0        embed       z = −14      z = −28            z = −168           z = −182
```

- **Residual stream**: for each token, a hairline rail along −z through the whole tower. This is the
  backbone; every block reads from it (a branch peels off) and writes back into it (an add).
- **Embedding bay** at z ≈ +6: the W_E slab stands to the left, W_P slab to the right.
- **Each block** occupies 14 units of depth: an Attention slab (purple) then an MLP slab (gold).
  The walkthrough opens block 1 (layer 0) and lays its internals out *beside* the stream
  (x offset) so the stream itself stays readable.
- **Head end** at z ≈ −182: LN_f, the W_U slab, and a fan of probability bars.

Level of detail: when the camera is far (> 40 units) a block renders as its glass slab plus a
heat-colored strip of its output norms; numbers appear only within ~25 units.

---

## 5. Storyboard

Each step lists: **Q** = the learner's question it answers · **Camera** · **Animation** (as a function
of p) · **Text** · **Shapes** shown. Demo prompt: *"The Eiffel Tower is in the city of"* (10 tokens),
which GPT-2 continues with " Paris".

### 0. Cold open
- **Q**: What am I looking at?
- **Camera**: starts far back and high, the whole 12-block tower in silhouette, dollies slowly in.
- **Animation**: the prompt types itself in serif text above the tower; at the end the word " Paris"
  appears faintly at the far end with "p = 0.xx".
- **Text**: "Every word GPT-2 writes is the output of the same pipeline. This is that pipeline, with
  the real numbers from your sentence."
- **Shapes**: 124,439,808 parameters, 12 layers.

### 1. Tokenization
- **Q**: Does the model read letters or words?
- **Camera**: front-on, close to the prompt text.
- **Animation**: p 0–0.4 the sentence splits at BPE boundaries (gaps open, each piece gets a glass
  pill); 0.4–0.8 each pill flips to show its integer id underneath; 0.8–1 pills settle into the
  token row at the top of the token columns. Spaces are drawn as a faint `·` because GPT-2 glues
  the leading space onto the word (" Tower" is one token, id 8765).
- **Text**: byte-level BPE in one paragraph: start from bytes, merge the most frequent pairs 50,000
  times during training; common words become one token, rare ones split ("Eiffel" → "E" + "iff" + "el").
- **Shapes**: `T = 10` tokens, ids ∈ [0, 50257).

### 2. The embedding matrix
- **Q**: How does an integer become something a network can compute with?
- **Camera**: pulls back to reveal W_E as a towering slab (its height dwarfs the screen), then
  pans down it as rows light up.
- **Animation**: p 0–0.25 slab fades in, edge labels "50,257 rows" and "768 columns"; 0.25–0.75 for each
  token in sequence, its id travels to the slab, a thin horizontal band lights at row = id (the
  band's height position is the true fraction id/50257), the 768 numbers of that row (shown as
  first 6, ⋮, last) lift off the slab, rotate 90° and fly to the token's column; 0.75–1 all T
  columns stand in a row, labeled.
- **Text**: "W_E is a lookup table: row *i* is the learned vector for token *i*. Multiplying a one-hot
  vector by W_E selects that row; the model just indexes it."
- **Math**: $x_i = W_E[t_i] \in \mathbb{R}^{768}$, $W_E \in \mathbb{R}^{50257\times768}$ (38.6M parameters).
- **Shapes**: W_E `[50257, 768]`, each column `[768]`, stacked `[T, 768]`.

### 3. Positional embeddings
- **Q**: "the cat ate the fish" and "the fish ate the cat" — how does the model know order?
- **Camera**: slight orbit to the right, W_P slab (amber, 1024 × 768) appears.
- **Animation**: row *i* of W_P (rows 0..T−1 light up top-down) flies into a second, amber column
  beside each token column; then the two columns slide together and each entry visibly sums
  (the white and amber digits merge, the result digits roll to the new values).
- **Text**: without this, attention is order-blind (a weighted sum doesn't care about order).
  GPT-2 *learns* one vector per position, up to 1,024.
- **Math**: $x^0_i = W_E[t_i] + W_P[i]$.
- **Shapes**: W_P `[1024, 768]`, x⁰ `[T, 768]`.

### 4. The residual stream
- **Q**: What happens to these vectors next — do they get replaced?
- **Camera**: rises and swings behind the columns to look down −z; the tower appears in depth.
- **Animation**: each column stretches into a glowing rail that runs through all 12 blocks. A pulse
  travels down each rail; at every block a branch peels off, passes through the block, and merges
  back with a "+". Rails brighten slightly after each add (real ‖x‖ growth per layer, from the trace).
- **Text**: the central mental model. Every layer *reads* from the stream and *adds* its result
  back; nothing overwrites. The stream is a shared 768-dimensional workspace per token.
- **Math**: $x^{L+1} = x^{L} + \text{Attn}(\text{LN}(x^L)) + \text{MLP}(\text{LN}(\cdot))$.
- **Shapes**: stream `[T, 768]` at every depth; small sparkline of ‖x_i‖ per layer.

### 5. LayerNorm
- **Q**: Why normalize before every sublayer?
- **Camera**: dives into block 1, one token's column at center.
- **Animation**: the 768 entries are shown as a strip of tick marks on a number line (all 768, not
  truncated). p 0–0.3 a marker shows the mean, the strip slides so the mean sits at 0 (centering);
  0.3–0.6 the strip squeezes/stretches to unit standard deviation (rescaling); 0.6–1 per-dimension
  gain γ and bias β nudge individual ticks; the column of numbers updates in sync.
- **Side diagram**: a 3D point cloud view — centering projects onto the plane ⟂ to (1,1,…,1),
  rescaling puts the point on a sphere of radius √768.
- **Text**: keeps every sublayer's input at a predictable scale no matter how large the stream has
  grown; the learned γ, β let the layer choose its own scale per dimension.
- **Math**: $\text{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$, real μ and σ shown.
- **Shapes**: x `[768]` → `[768]`; γ, β `[768]`.

### 6. Queries, keys and values
- **Q**: How does a token decide what to look for and what to offer?
- **Camera**: wide enough for three slabs side by side above the normalized columns.
- **Animation**: three slabs W_Q (blue), W_K (red), W_V (green), each `768 × 768`. One token's column
  rotates into a row, slides across the slab (a real matrix–vector product: a highlight sweeps the
  slab's rows while the output entry being computed glows), and the output column drops out below,
  tinted. Repeat quickly for all tokens. Then each 768-column splits into 12 stacked segments of 64
  with hairline gaps and labels h1…h12.
- **Text**: Q = what this token is looking for; K = what it advertises; V = what it hands over if
  chosen. All three are just linear maps of the same normalized vector.
- **Math**: $q_i = a_i W_Q + b_Q$ etc.; stored as one `768×2304` matrix.
- **Shapes**: W_Q,W_K,W_V `[768,768]` each; Q,K,V `[T,768]` → `[12,T,64]`.

### 7. Attention scores (dot products)
- **Q**: How does one token measure relevance to another?
- **Camera**: focus on head h (the most interesting head of block 1, chosen by pattern entropy),
  the last token's query on the left, keys of all tokens in a row.
- **Animation**: q_T (blue, 64 entries) slides over each k_j in turn; paired entries multiply
  (small products appear) and sum into one number that drops into a score row. Then the full T×T
  grid fills row by row.
- **Side panel — the dot product**: two arrows a and b in 3D with a draggable angle.
  $a\cdot b = \lVert a\rVert\lVert b\rVert\cos\theta$. cos θ measures alignment (direction match); the
  lengths measure how strongly. Attention uses the *unnormalized* dot product on purpose: a key with a
  large norm can shout, and a query can be sharp (large) or vague (small) — normalizing would throw
  away that confidence signal.
- **Shapes**: q `[64]`, K_h `[T,64]`, scores `[T,T]` per head.

### 8. Scale, mask, softmax
- **Q**: How do raw scores become weights, and how can the model avoid peeking at the future?
- **Camera**: the T×T grid fills the frame, slight tilt.
- **Animation**: p 0–0.3 every cell divides by 8 (√64), numbers roll down; 0.3–0.6 the upper triangle
  (j > i) turns to −∞ (cells go dark with a "−∞" glyph, a glass sheet slides over them); 0.6–1 each
  row exponentiates and normalizes: cells become heat squares, row sums "= 1" appear at right.
- **Text**: dividing by √d_head keeps the variance of scores ≈ 1 so softmax doesn't saturate. The
  mask makes position i see only 1..i, which is what makes training on all positions at once honest.
- **Math**: $A=\text{softmax}\big(\frac{QK^\top}{\sqrt{64}} + M\big)$, $M_{ij}=-\infty$ for $j>i$.
- **Shapes**: `[T,T]`, each row sums to 1.

### 9. Weighted sum of values
- **Q**: Once it knows where to look, what does a token actually receive?
- **Camera**: back to columns: value vectors (green) above each token.
- **Animation**: for the focused token, glowing green particles flow from every earlier value vector
  to it, particle density ∝ attention weight; the received vector assembles (each entry is a
  weighted sum; the top contributor's share is labeled).
- **Math**: $z_i=\sum_{j\le i} A_{ij}\, v_j$.
- **Shapes**: A row `[T]`, V_h `[T,64]`, z_i `[64]`.

### 10. Many heads, then W_O
- **Q**: Why twelve heads, and how do their results combine?
- **Camera**: pulls back: 12 small T×T grids in a 4×3 arc, each the *real* pattern of that head.
- **Animation**: a caption under each head guesses its job from its pattern (previous-token,
  first-token sink, self, broad) — computed, not hand-written. Then 12 z-segments (64 each)
  concatenate into one 768 column, pass through W_O (purple slab), and the output drops into the
  residual rail with a "+": the rail's numbers roll to x + attn_out.
- **Math**: $\text{attn\_out} = [z^1;\dots;z^{12}]\,W_O + b_O$.
- **Shapes**: 12 × `[T,64]` → `[T,768]`; W_O `[768,768]`.

### 11. The MLP
- **Q**: Attention moved information between tokens. What does each token do with it?
- **Camera**: single token, the column widens to the right.
- **Animation**: after LN₂, the 768 column passes W_up (gold slab 768 × 3072, visibly 4× wider),
  expanding into a *wide* field of 3072 neurons (laid out 48 × 64, real pre-activations as dots);
  GELU pass: each neuron dims or lights according to GELU(x) (the negatives collapse toward 0);
  the GELU curve is drawn beside it with the real points scattered on it. Then W_down compresses back
  to 768 and adds into the stream.
- **Text**: the MLP works per token, no mixing. It is where most parameters live (2/3 of each
  block) and where much factual association is stored.
- **Math**: $\text{GELU}(x)\approx 0.5x(1+\tanh[\sqrt{2/\pi}(x+0.044715x^3)])$.
- **Shapes**: `[768] → [3072] → [768]`; W_up `[768,3072]`, W_down `[3072,768]`; % of neurons > 0.

### 12. Repeat ×12
- **Q**: Is that it?
- **Camera**: long pull back along −z until all 12 blocks stack in depth, then a slow orbit.
- **Animation**: the stream pulses through all blocks; each block shows a thin strip of its
  real attention-head patterns and MLP firing rate. Label: "Many repetitions — 12 blocks, same
  shape, different weights".
- **Shapes**: 12 × (Attn + MLP), 7.1M params per block, 85M in blocks.

### 13. Final LayerNorm and unembedding
- **Q**: How does a 768-vector become a word?
- **Camera**: arrives at the head end; W_U slab (same physical slab as W_E, rotated — it's tied).
- **Animation**: only the **last** column continues; LN_f; the vector slides along the 50,257 rows of
  W_U and a column of 50,257 logits appears (shown as a tall bar strip with the top 10 labeled).
- **Text**: W_U = W_Eᵀ — the same table maps tokens in and scores tokens out.
- **Math**: $\ell = \text{LN}_f(x^{12}_T)\,W_E^\top$.
- **Shapes**: `[768] × [768,50257] → [50257]`.

### 14. Softmax, temperature, top-k, top-p, sampling
- **Q**: How does a list of scores become one chosen word?
- **Camera**: the logit strip rotates into a horizontal bar chart of the top 20.
- **Animation**: softmax turns bars into probabilities; a temperature dial animates through
  0.5 → 1 → 1.5 (distribution sharpens and flattens, real values); top-k cuts the tail; top-p
  draws the cumulative line and cuts at 0.9; a ball rolls along the cumulative bar and stops on the
  sampled token. Interactive in place.
- **Math**: $p_v=\frac{e^{\ell_v/\tau}}{\sum_u e^{\ell_u/\tau}}$.

### 15. Autoregression and the KV cache
- **Q**: How does it write a whole paragraph?
- **Camera**: back to the front of the tower, slightly elevated.
- **Animation**: " Paris" becomes a new token, gets embedded, and only its column runs through the
  tower; the old K and V vectors of every layer sit in glass trays (the cache) and light up as they
  are *read*, not recomputed. A counter: "computed this step: 1 column; reused: T columns × 12 layers".
- **Text**: because of the causal mask, earlier rows never change when a token is appended, which
  is exactly why caching them is exact, not an approximation.

### 16. Training
- **Q**: Where do all these numbers come from?
- **Camera**: wide, all T columns at the head end.
- **Animation**: every column (not just the last) produces a distribution; arrows connect position i
  to the actual token i+1 with its real probability and loss −log p; the losses average into one
  number. The mask flashes to show why this isn't cheating. Then a gradient pulse runs backwards.
- **Text**: training = every position predicts the next one, all at once. At inference only the last
  prediction is sampled — but earlier positions still matter because later tokens attend to their
  keys and values.
- **Math**: $\mathcal{L}=-\frac1{T-1}\sum_{i=1}^{T-1}\log p_\theta(t_{i+1}\mid t_{\le i})$.

---

## 6. Architecture

```
src/
  model/            pure TS, no DOM — runs in a Web Worker and in Node (tests, demo build)
    tokenizer.ts    byte-level BPE (regex pre-tokenizer, byte↔unicode map, merge ranks)
    safetensors.ts  header parse, Range-request planning, typed views
    gpt2.ts         forward(ids, cache) → Trace; CPU kernels; optional GPU backend
    webgpu.ts       WGSL tiled matmul for the linear layers and the unembedding
    sample.ts       temperature / top-k / top-p / sampling (seeded RNG)
    trace.ts        Trace type, binary (de)serialization, merge for incremental steps
  worker/engine.worker.ts   load weights (progress, Cache Storage), run forward, post Trace
  scene/
    glyphs/         atlas generation + NumberField instanced mesh (from/to flight attributes)
    primitives/     GlassSlab, VectorColumn, MatrixFace, Rail, AttnGrid, Bars, MathLabel
    steps/          one component per storyboard step; each is f(progress, trace)
    camera.ts       global keyframe path in story space; damped follow
    world.ts        coordinates of every object (single source of truth)
    explore/        tower overview, picking, layer scrubber, sampling panel
  story/            step metadata + prose (MD + KaTeX) + per-step shapes
  ui/               scroll column, HUD, prompt box, loading state, reduced-motion toggle
```

**Why these choices.**
- *React Three Fiber + drei*: declarative scene graph that maps 1:1 to the storyboard's objects.
- *Own choreography instead of GSAP/Theatre.js*: every step is a pure function of scroll progress,
  which makes scrubbing backwards exact, makes the logic unit-testable, and avoids a second
  timeline system fighting the scroll. A critically damped spring smooths the progress value.
- *Own instanced glyph renderer instead of troika for numbers*: troika creates one mesh per text
  block and re-lays out on change; we need ~20k digits that fly independently. One atlas + one
  instanced quad mesh per vector is ~100× cheaper. troika (drei `Text`) is still fine for the few
  large labels, and KaTeX DOM labels (drei `Html`) handle all math.
- *postprocessing*: a single EffectComposer pass (bloom + vignette + SMAA). Depth of field is
  optional and off by default on low frame rates.
- *Own forward pass*: needed to expose every intermediate; transformers.js hides them.

**Data flow.**
1. Page loads → `demo.trace.bin` (precomputed by `scripts/build-demo.ts` with the same engine) →
   scene renders immediately.
2. Tokenizer files load (1.5 MB) → typing shows live tokens.
3. User opts in (or explore mode) → worker downloads the 124M weights via Range requests
   (≈ 498 MB fp32, cached in Cache Storage for next time) with a progress bar.
4. `forward(ids)` → `Trace` (all intermediates, Float32Array) → transferred to main thread →
   `TraceContext` → every step component reads from it.
5. Generate: worker runs incremental `forward([new], cache)`; trace rows are appended (old rows
   are unchanged because of the causal mask), and the scene animates the new column.

**Trace contents** per prompt (T tokens): token ids/strings; `embedTok, embedPos, resid[0..12]`
`[T,768]`; per layer `ln1, q, k, v, z, attnOut, residMid, ln2, mlpOut [T,768]`, `scores, pattern
[12,T,T]`, `mlpPre, mlpPost [T,3072]`, LN statistics; `lnF [T,768]`; `logitsLast [50257]`;
per position: top-20 ids and probabilities plus the probability of the true next token; weight
previews (corners of each displayed matrix, plus the exact rows used).

**Backends.** CPU kernel (`i`-outer loop, contiguous inner `o` loop; V8 vectorizes it) is the
reference. WebGPU kernel (16×16 tiles, shared-memory) handles `[T,768]×[768,N]` and the
`[T,768]×[50257,768]ᵀ` unembedding; both backends are tested against the HF fixture.

---

## 7. Quality gates

- `npm test`: tokenizer vs HF on 13 strings (exact ids); forward vs HF fixture (3 prompts, every
  stored intermediate, max abs error < 2e-3 on activations, < 5e-3 on logits, identical top-1);
  KV-cache incremental equals full recompute; sampling statistics; story/camera continuity
  (camera path is continuous at every step boundary; every step declares shapes).
- `npm run shots`: Playwright renders every step at p = 0.2, 0.5, 0.9 to `shots/` for review.
- Performance: frame time logged in dev HUD; target 16.7 ms on an M-series laptop.
- Reduced motion: `prefers-reduced-motion` or toggle → transitions become 150 ms fades, camera
  jumps between step key poses, no flying digits.
- Mobile (< 768 px or no WebGL2): stacked text with one static rendered frame per step; explore
  shows a 2D attention/probability view on the precomputed demo.
