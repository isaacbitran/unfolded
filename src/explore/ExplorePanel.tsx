// Explore mode controls: depth scrubber, attention-head inspector, and the sampling playground.

import { useEffect, useMemo, useRef, useState } from 'react';
import { generate } from '../engine/client';
import { distribution } from '../model/sample';
import { useApp } from '../state/store';

const show = (s: string) => s.replace(/^ /, '·').replace(/\n/g, '↵');

export function ExplorePanel() {
  const trace = useApp((s) => s.trace)!;
  const focus = useApp((s) => s.focus);
  const sampling = useApp((s) => s.sampling);
  const engine = useApp((s) => s.engine);
  const setFocus = useApp((s) => s.setFocus);
  const setSampling = useApp((s) => s.setSampling);
  const layer = Math.max(1, focus.layer);
  const T = trace.T;
  const busy = engine.kind === 'loading' || engine.kind === 'running';
  const cands = useMemo(() => distribution(trace.logitsLast, sampling, 14), [trace, sampling]);
  const text = (id: number) => trace.vocabText[id] ?? `#${id}`;

  return (
    <aside className="explore">
      <section>
        <h3>Residual stream depth</h3>
        <input type="range" min={0} max={12} value={focus.layer} onChange={(e) => setFocus({ layer: +e.target.value })} aria-label="Depth" />
        <div className="ticks">
          <span>embed</span>
          <span>{focus.layer === 0 ? 'x⁰ = embeddings' : `after block ${focus.layer}`}</span>
          <span>12</span>
        </div>
        <p className="hint">Drag to orbit · scroll to zoom · click a block to fly to it.</p>
      </section>

      <section>
        <h3>
          Attention · block {layer}, head {focus.head + 1}
        </h3>
        <div className="heads">
          {Array.from({ length: 12 }, (_, h) => (
            <HeadThumb key={h} pattern={trace.layers[layer - 1].pattern} h={h} T={T} on={h === focus.head} onClick={() => setFocus({ head: h, layer })} />
          ))}
        </div>
        <AttnGrid pattern={trace.layers[layer - 1].pattern} h={focus.head} tokens={trace.tokens.map((t) => show(t.text))} />
      </section>

      <section>
        <h3>Next token</h3>
        <Slider label="temperature" value={sampling.temperature} min={0} max={2} step={0.05} onChange={(v) => setSampling({ temperature: v })} />
        <Slider label="top-k" value={sampling.topK} min={0} max={100} step={1} onChange={(v) => setSampling({ topK: v })} fmt={(v) => (v ? String(v) : 'off')} />
        <Slider label="top-p" value={sampling.topP} min={0.05} max={1} step={0.01} onChange={(v) => setSampling({ topP: v })} fmt={(v) => (v >= 1 ? 'off' : v.toFixed(2))} />
        <ol className="dist">
          {cands.map((c) => (
            <li key={c.id} className={c.kept > 0 ? '' : 'cut'}>
              <span className="tok">{show(text(c.id))}</span>
              <span className="bar">
                <span className="p" style={{ width: `${Math.min(100, c.prob * 100 * 2.2)}%` }} />
                <span className="k" style={{ width: `${Math.min(100, c.kept * 100 * 2.2)}%` }} />
              </span>
              <span className="num">{(c.kept * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ol>
        <div className="gen">
          <button disabled={busy} onClick={() => generate(1).catch(() => undefined)}>Generate 1</button>
          <button disabled={busy} onClick={() => generate(10).catch(() => undefined)}>Generate 10</button>
        </div>
        <p className="hint">Bars: light = softmax(ℓ/τ) over all 50,257 tokens; bright = after top-k / top-p, renormalised. Generating loads the real model.</p>
      </section>
    </aside>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <label className="slider">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
      <span className="v">{fmt ? fmt(value) : value.toFixed(2)}</span>
    </label>
  );
}

function HeadThumb({ pattern, h, T, on, onClick }: { pattern: Float32Array; h: number; T: number; on: boolean; onClick: () => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current!;
    const ctx = c.getContext('2d')!;
    const s = c.width / T;
    ctx.clearRect(0, 0, c.width, c.height);
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        const a = pattern[(h * T + i) * T + j];
        ctx.fillStyle = `rgba(180,140,255,${0.08 + 0.92 * Math.sqrt(a)})`;
        ctx.fillRect(j * s, i * s, s - 0.5, s - 0.5);
      }
  }, [pattern, h, T]);
  return (
    <button className={`thumb ${on ? 'on' : ''}`} onClick={onClick} title={`head ${h + 1}`}>
      <canvas ref={ref} width={44} height={44} />
      <span>{h + 1}</span>
    </button>
  );
}

function AttnGrid({ pattern, h, tokens }: { pattern: Float32Array; h: number; tokens: string[] }) {
  const T = tokens.length;
  const [hover, setHover] = useState<[number, number] | null>(null);
  const cell = Math.max(14, Math.min(30, Math.floor(250 / T)));
  return (
    <div className="grid-wrap">
      <div className="grid" style={{ gridTemplateColumns: `auto repeat(${T}, ${cell}px)` }} onMouseLeave={() => setHover(null)}>
        <span />
        {tokens.map((t, j) => (
          <span key={j} className="col-l" style={{ height: 64 }}>
            <span>{t}</span>
          </span>
        ))}
        {tokens.map((t, i) => (
          <Row key={i} i={i} t={t} T={T} cell={cell} pattern={pattern} h={h} hover={hover} setHover={setHover} />
        ))}
      </div>
      <div className="readout">
        {hover ? (
          <>
            <b>{tokens[hover[0]]}</b> → <b>{tokens[hover[1]]}</b>: {hover[1] > hover[0] ? 'masked (future)' : pattern[(h * T + hover[0]) * T + hover[1]].toFixed(3)}
          </>
        ) : (
          'rows: query token · columns: key token · hover a cell'
        )}
      </div>
    </div>
  );
}

function Row({ i, t, T, cell, pattern, h, hover, setHover }: { i: number; t: string; T: number; cell: number; pattern: Float32Array; h: number; hover: [number, number] | null; setHover: (v: [number, number]) => void }) {
  return (
    <>
      <span className={`row-l ${hover?.[0] === i ? 'on' : ''}`}>{t}</span>
      {Array.from({ length: T }, (_, j) => {
        const a = j > i ? 0 : pattern[(h * T + i) * T + j];
        return (
          <span
            key={j}
            className={`cell ${j > i ? 'masked' : ''}`}
            style={{ width: cell, height: cell, background: j > i ? undefined : `rgba(180,140,255,${0.06 + 0.94 * Math.sqrt(a)})` }}
            onMouseEnter={() => setHover([i, j])}
          />
        );
      })}
    </>
  );
}
