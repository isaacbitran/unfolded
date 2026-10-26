// The walkthrough: one entry per storyboard step (DESIGN.md §5). Prose uses $…$ for KaTeX.
// Camera keys are in story space: key.p ∈ [0,1] is progress within the step.

export type V3 = [number, number, number];
export interface CamKey {
  p: number;
  pos: V3;
  look: V3;
  fov?: number;
}

export interface Step {
  id: string;
  title: string;
  /** the question a first-time learner is asking at this point */
  question: string;
  body: string;
  /** tensor shapes involved, KaTeX */
  shapes: string[];
  cam: CamKey[];
}

export const STEPS: Step[] = [
  {
    id: 'intro',
    title: 'Unfolded',
    question: 'What happens between typing a sentence and GPT-2 answering?',
    body: `Every word GPT-2 writes comes out of the same pipeline: a few hundred matrix multiplications and a handful of simple functions, repeated in 12 layers.

This page walks through that pipeline **one operation at a time**, using GPT-2 small running in your browser. Every number you see is a real value from the model, computed on the sentence above.

Scroll, or use the arrow keys.`,
    shapes: ['124{,}439{,}808\\text{ parameters}', '12\\text{ layers}', 'd_{\\text{model}}=768'],
    cam: [
      { p: 0, pos: [26, 24, 60], look: [0, -2, -80], fov: 32 },
      { p: 0.85, pos: [10, 6, 22], look: [0, 0, -40], fov: 34 },
    ],
  },
  {
    id: 'tokens',
    title: '1 · Tokenization',
    question: 'Does the model read letters, or words?',
    body: `Neither, exactly. GPT-2 reads **tokens**: chunks of text from a fixed vocabulary of 50,257 pieces.

The vocabulary was built by *byte-pair encoding*: start from the 256 possible bytes, then repeatedly merge the most frequent adjacent pair in a large corpus, 50,000 times. Common words end up as one token; rare words split into pieces ("Eiffel" → "E" + "iff" + "el").

A leading space is part of the token, so " Jordan" and "Jordan" are different tokens. Each token is now just an integer: its row number in the vocabulary.`,
    shapes: ['T = \\text{number of tokens}', 't_i \\in \\{0,\\dots,50256\\}'],
    cam: [
      { p: 0, pos: [0, 3.2, 15], look: [0, 2.2, 0], fov: 34 },
      { p: 1, pos: [0, 2.6, 14], look: [0, 1.6, 0], fov: 34 },
    ],
  },
  {
    id: 'embed',
    title: '2 · The embedding matrix',
    question: 'How does an integer become something a network can compute with?',
    body: `GPT-2 keeps a table $W_E$ with one row per token: 50,257 rows, each a list of 768 learned numbers.

Token $t_i$ simply **picks out its row**. (Multiplying a one-hot vector by $W_E$ would give the same result; the model just indexes.) That row becomes the token's vector — drawn here as a column, one per token.

$$x_i = W_E[t_i] \\in \\mathbb{R}^{768}$$

Those 768 numbers are where the model starts building its idea of what the token means. $W_E$ alone holds 38.6 million of GPT-2's 124 million parameters.`,
    shapes: ['W_E \\in \\mathbb{R}^{50257\\times 768}', 'x_i \\in \\mathbb{R}^{768}', 'X \\in \\mathbb{R}^{T\\times 768}'],
    cam: [
      { p: 0, pos: [-6, -16, 26], look: [-10, -11, 0], fov: 34 },
      { p: 0.16, pos: [-5, -1, 29], look: [-5.5, -0.8, 0], fov: 34 },
      { p: 0.72, pos: [-3.5, 0, 27], look: [-4, -0.4, 0], fov: 34 },
      { p: 1, pos: [0, 0.4, 16.5], look: [0, 0.2, 0], fov: 34 },
    ],
  },
  {
    id: 'position',
    title: '3 · Positional embeddings',
    question: '"Dog bites man" and "man bites dog" — how does the model know the order?',
    body: `So far nothing says *where* a token is. Attention, coming up, mixes tokens with weighted sums — and a sum doesn't care about order.

GPT-2 fixes this with a second table $W_P$: one learned vector per position, 0 through 1,023. Position $i$'s vector is **added** to token $i$'s vector, entry by entry.

$$x^{0}_i = W_E[t_i] + W_P[i]$$

The same word now looks slightly different at different positions. That is also why GPT-2 can't read more than 1,024 tokens: there are no rows beyond that.`,
    shapes: ['W_P \\in \\mathbb{R}^{1024\\times 768}', 'X^{0} \\in \\mathbb{R}^{T\\times 768}'],
    cam: [
      { p: 0, pos: [2, 0.6, 17], look: [1, 0, 0], fov: 34 },
      { p: 1, pos: [0, 0.4, 16.5], look: [0, 0, 0], fov: 34 },
    ],
  },
  {
    id: 'residual',
    title: '4 · The residual stream',
    question: 'Do the layers replace these vectors?',
    body: `No — and this is the most useful idea for understanding transformers.

Each token's vector becomes a **stream** that flows through all 12 layers. Every layer *reads* the stream, computes something, and **adds** its result back. Nothing is overwritten.

$$x^{\\ell+1} = x^{\\ell} + \\text{Attn}(\\cdot) + \\text{MLP}(\\cdot)$$

So the stream is a shared 768-dimensional workspace per token. Attention writes in information gathered from *other* tokens; the MLP writes in what it computes from *this* token. The brightness of each rail tracks the real size $\\lVert x \\rVert$ of the stream as it grows layer by layer.`,
    shapes: ['x^{\\ell} \\in \\mathbb{R}^{T\\times 768}\\ \\text{at every depth}'],
    cam: [
      { p: 0, pos: [14, 9, 14], look: [0, -1, -30], fov: 36 },
      { p: 1, pos: [16, 12, 6], look: [0, -2, -70], fov: 38 },
    ],
  },
  {
    id: 'layernorm',
    title: '5 · LayerNorm',
    question: 'Why normalize before every sublayer?',
    body: `As layers keep adding into it, the stream's numbers drift and grow. LayerNorm gives each sublayer an input with a predictable scale.

For one token's 768 numbers it **subtracts their mean** (centering) and **divides by their standard deviation** (rescaling). Geometrically the vector is projected onto the plane perpendicular to $(1,1,\\dots,1)$ and then placed on a sphere of radius $\\sqrt{768}\\approx 27.7$.

Then two learned vectors give back control: a gain $\\gamma$ and a bias $\\beta$ per dimension.

$$\\text{LN}(x)=\\gamma\\odot\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}+\\beta$$`,
    shapes: ['x\\in\\mathbb{R}^{768}\\to\\mathbb{R}^{768}', '\\gamma,\\beta\\in\\mathbb{R}^{768}'],
    cam: [
      { p: 0, pos: [0, 0.4, 13], look: [0, 0, 0], fov: 34 },
      { p: 1, pos: [0, 0.2, 12], look: [0, 0, 0], fov: 34 },
    ],
  },
  {
    id: 'qkv',
    title: '6 · Queries, keys and values',
    question: 'How does a token decide what to look for and what to offer?',
    body: `Each normalized vector is multiplied by three matrices, giving three new vectors per token:

- a **query** $q_i = a_i W_Q$: what this token is looking for,
- a **key** $k_i = a_i W_K$: what this token can be found by,
- a **value** $v_i = a_i W_V$: what this token hands over if it is chosen.

Each is a real matrix–vector product: every output entry is the dot product of the input with one column of the matrix.

The 768 entries are then cut into **12 heads of 64**. Each head runs attention independently with its own slice of $Q$, $K$ and $V$.`,
    shapes: ['W_Q,W_K,W_V\\in\\mathbb{R}^{768\\times768}', 'Q,K,V\\in\\mathbb{R}^{T\\times768}\\to\\mathbb{R}^{12\\times T\\times 64}'],
    cam: [
      { p: 0, pos: [0, 1.5, 22], look: [0, 0.5, 0], fov: 34 },
      { p: 1, pos: [0, 1, 20], look: [0, 0, 0], fov: 34 },
    ],
  },
  {
    id: 'scores',
    title: '7 · Attention scores',
    question: 'How does one token measure how relevant another token is?',
    body: `With a dot product. For one head, the last token's query $q$ is compared with the key $k_j$ of every token so far:

$$s_j = q\\cdot k_j = \\sum_{d=1}^{64} q_d\\,k_{j,d}$$

Geometrically $q\\cdot k = \\lVert q\\rVert\\,\\lVert k\\rVert\\cos\\theta$. The angle part measures **alignment**: does the key point the way the query is looking? The lengths measure **strength**: how strongly it matches.

Attention deliberately uses the *unnormalized* dot product. A head can make a query long to be decisive or short to spread its attention, and a key can be long to stand out — normalizing would throw that signal away.`,
    shapes: ['q\\in\\mathbb{R}^{64}', 'K_h\\in\\mathbb{R}^{T\\times64}', 'S_h=Q_hK_h^{\\top}\\in\\mathbb{R}^{T\\times T}'],
    cam: [
      { p: 0, pos: [0, 0.5, 16], look: [0, 0, 0], fov: 34 },
      { p: 1, pos: [0, 0.5, 15], look: [0, 0, 0], fov: 34 },
    ],
  },
  {
    id: 'softmax',
    title: '8 · Scale, mask, softmax',
    question: 'How do raw scores become weights — and how does the model avoid peeking at the future?',
    body: `Three steps turn the score grid into attention weights:

1. **Scale** by $1/\\sqrt{64} = 1/8$. Dot products of 64-dimensional vectors grow with dimension; scaling keeps them in a range where softmax still has gradients.
2. **Mask**: every score where a token would look at a *later* token is set to $-\\infty$. Token $i$ can only see tokens $1..i$.
3. **Softmax** each row: $e^{s}$ then divide by the row total, so each row is a set of positive weights summing to 1.

$$A=\\text{softmax}\\Big(\\tfrac{QK^{\\top}}{\\sqrt{64}}+M\\Big)$$`,
    shapes: ['A_h\\in\\mathbb{R}^{T\\times T}', '\\textstyle\\sum_j A_{ij}=1'],
    cam: [
      { p: 0, pos: [0, 0.3, 14], look: [0, 0, 0], fov: 34 },
      { p: 1, pos: [0, 0.3, 13.5], look: [0, 0, 0], fov: 34 },
    ],
  },
  {
    id: 'values',
    title: '9 · Weighted sum of values',
    question: 'Once it knows where to look, what does a token actually receive?',
    body: `A blend of value vectors. Token $i$ receives each earlier token's value, scaled by how much it attends to it, all added up:

$$z_i=\\sum_{j\\le i}A_{ij}\\,v_j$$

If a head puts 0.8 of its attention on " Jordan", then 80% of what this head delivers is " Jordan"'s value vector. This is the **only** place in the whole model where information moves between tokens.`,
    shapes: ['A_{i,:}\\in\\mathbb{R}^{T}', 'V_h\\in\\mathbb{R}^{T\\times64}', 'z_i\\in\\mathbb{R}^{64}'],
    cam: [
      { p: 0, pos: [0, 1, 17], look: [0, 0, 0], fov: 34 },
      { p: 1, pos: [0, 1, 16], look: [0, 0, 0], fov: 34 },
    ],
  },
  {
    id: 'heads',
    title: '10 · Twelve heads, then W_O',
    question: 'Why twelve heads, and how do their results combine?',
    body: `Each head has its own $W_Q, W_K, W_V$ slices, so each learns to look for something different. These are the real attention patterns of all 12 heads in layer 1 for your sentence; the captions are computed from the patterns.

The 12 outputs (64 numbers each) are **concatenated** back into 768 numbers and multiplied by one more matrix, $W_O$, which mixes the heads together. The result is **added** to the residual stream.

$$\\text{attn}(X) = [Z^1;\\dots;Z^{12}]\\,W_O + b_O$$`,
    shapes: ['12\\times(T\\times64)\\to T\\times768', 'W_O\\in\\mathbb{R}^{768\\times768}'],
    cam: [
      { p: 0, pos: [0, 2, 26], look: [0, 1, 0], fov: 34 },
      { p: 1, pos: [0, 1, 22], look: [0, 0, 0], fov: 34 },
    ],
  },
  {
    id: 'mlp',
    title: '11 · The MLP',
    question: 'Attention moved information between tokens. What does each token do with it?',
    body: `It thinks on its own. The MLP processes each token **separately** — no mixing between tokens here.

After another LayerNorm the 768 numbers are projected **up** to 3,072 "neurons", passed through the GELU nonlinearity, and projected back **down** to 768:

$$\\text{MLP}(a)=\\text{GELU}(a\\,W_{\\uparrow}+b_{\\uparrow})\\,W_{\\downarrow}+b_{\\downarrow}$$

GELU lets positive inputs through and squashes negative ones toward zero, so each neuron is roughly *on* or *off* for this token. Two thirds of every layer's parameters live here, and much of what the model "knows" (Michael Jordan → basketball) is stored in these weights.`,
    shapes: ['768\\to3072\\to768', 'W_{\\uparrow}\\in\\mathbb{R}^{768\\times3072}', 'W_{\\downarrow}\\in\\mathbb{R}^{3072\\times768}'],
    cam: [
      { p: 0, pos: [0, 0.5, 18], look: [0, 0, 0], fov: 34 },
      { p: 1, pos: [0, 0.5, 20], look: [0, 0, 0], fov: 34 },
    ],
  },
  {
    id: 'stack',
    title: '12 · Many repetitions',
    question: 'Is that the whole model?',
    body: `That was one block: attention, then MLP, each wrapped in a LayerNorm and a residual add.

GPT-2 small stacks **12 identical blocks**, each with its own weights (7.1 million parameters per block). The stream flows through all of them, and every block adds a little more: early layers handle syntax and nearby words, later ones handle meaning and the prediction itself.`,
    shapes: ['12\\times(\\text{Attn}+\\text{MLP})', '85\\text{M parameters in blocks}'],
    cam: [
      { p: 0, pos: [18, 10, 14], look: [0, 0, -40], fov: 36 },
      { p: 1, pos: [42, 22, -60], look: [0, -2, -95], fov: 40 },
    ],
  },
  {
    id: 'unembed',
    title: '13 · From vector to vocabulary',
    question: 'How does a 768-number vector become a word?',
    body: `Only the **last** token's vector is used to predict what comes next. After a final LayerNorm, it is compared with every row of the embedding matrix — the same $W_E$ from the start, reused (GPT-2 *ties* the input and output tables):

$$\\ell = \\text{LN}_f(x^{12}_T)\\,W_E^{\\top}$$

That is 50,257 dot products, one per vocabulary entry. Each result, a **logit**, scores how well that token fits next.`,
    shapes: ['x^{12}_T\\in\\mathbb{R}^{768}', 'W_E^{\\top}\\in\\mathbb{R}^{768\\times50257}', '\\ell\\in\\mathbb{R}^{50257}'],
    cam: [
      { p: 0, pos: [10, 3, -170], look: [0, 0, -186], fov: 36 },
      { p: 1, pos: [4, 1, -175], look: [0, 0, -190], fov: 34 },
    ],
  },
  {
    id: 'sampling',
    title: '14 · Choosing a token',
    question: 'How does a list of scores become one chosen word?',
    body: `Softmax turns logits into probabilities. **Temperature** $\\tau$ divides the logits first: below 1 the distribution sharpens toward the favorite, above 1 it flattens.

$$p_v=\\frac{e^{\\ell_v/\\tau}}{\\sum_u e^{\\ell_u/\\tau}}$$

**Top-k** keeps only the $k$ most likely tokens. **Top-p** keeps the smallest set whose probabilities add up to $p$. Whatever survives is renormalized, and one token is drawn at random.`,
    shapes: ['p\\in\\mathbb{R}^{50257},\\ \\textstyle\\sum p=1'],
    cam: [
      { p: 0, pos: [3, 0.5, -178], look: [3, 0, -192], fov: 34 },
      { p: 1, pos: [3, 0.5, -178], look: [3, 0, -192], fov: 34 },
    ],
  },
  {
    id: 'autoregress',
    title: '15 · One token at a time',
    question: 'How does it write a whole paragraph?',
    body: `The chosen token is appended to the input and the whole process runs again — embed, 12 blocks, unembed, sample — once per new token.

Because of the causal mask, earlier tokens can never be affected by later ones, so their keys and values in every layer never change. GPT-2 **caches** them (the KV cache) and only computes the new token's column. Each new step costs one column instead of the whole sequence.`,
    shapes: ['\\text{new: }1\\times768', '\\text{cached: }12\\times2\\times T\\times768'],
    cam: [
      { p: 0, pos: [14, 8, 16], look: [2, 0, -20], fov: 36 },
      { p: 1, pos: [16, 9, 10], look: [2, 0, -40], fov: 36 },
    ],
  },
  {
    id: 'training',
    title: '16 · Where the numbers come from',
    question: 'How did GPT-2 learn all of this?',
    body: `By predicting the next token, at **every position at once**. Each column produces its own guess for the token that follows it, and the training loss is the average surprise $-\\log p$ of the true next tokens:

$$\\mathcal{L}=-\\frac{1}{T-1}\\sum_{i=1}^{T-1}\\log p_\\theta(t_{i+1}\\mid t_{\\le i})$$

The causal mask is what makes this honest: position $i$ cannot see token $i+1$, so it can't cheat.

At inference only the last position's prediction is used. The earlier positions still matter: every later token reads their keys and values.`,
    shapes: ['\\ell\\in\\mathbb{R}^{T\\times50257}', '\\mathcal{L}\\in\\mathbb{R}'],
    cam: [
      { p: 0, pos: [0, 4, -168], look: [0, 0, -186], fov: 38 },
      { p: 1, pos: [0, 5, -165], look: [0, 0, -186], fov: 40 },
    ],
  },
];
