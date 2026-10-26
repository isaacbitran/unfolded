import { useEffect, useRef, useState } from 'react';
import { WEIGHTS_MB, getTokenizer, runPrompt } from '../engine/client';
import type { Token } from '../model/tokenizer';
import { useApp } from '../state/store';

const show = (s: string) => s.replace(/^ /, '·').replace(/\n/g, '↵');

/** Prompt editor with live BPE tokenization, the model run button and the engine status. */
export function PromptBar() {
  const trace = useApp((s) => s.trace);
  const engine = useApp((s) => s.engine);
  const backend = useApp((s) => s.backendName);
  const ms = useApp((s) => s.lastRunMs);
  const [text, setText] = useState('');
  const [editing, setEditing] = useState(false);
  const [tokens, setTokens] = useState<Token[]>([]);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (trace && !editing) setText(trace.prompt);
  }, [trace, editing]);
  useEffect(() => {
    let live = true;
    getTokenizer().then((t) => live && setTokens(t.tokenize(text).slice(0, 80)));
    return () => void (live = false);
  }, [text]);

  const busy = engine.kind === 'loading' || engine.kind === 'running';
  const submit = () => {
    if (!text.trim() || busy) return;
    input.current?.blur();
    setEditing(false);
    runPrompt(text).catch(() => undefined);
  };

  let status: React.ReactNode = null;
  if (engine.kind === 'loading') {
    const pct = engine.total ? engine.loaded / engine.total : 0;
    status = (
      <span className="status">
        <span className="bar"><span style={{ width: `${(pct * 100).toFixed(1)}%` }} /></span>
        loading GPT-2 · {(engine.loaded / 1e6).toFixed(0)} / {(engine.total / 1e6).toFixed(0)} MB
      </span>
    );
  } else if (engine.kind === 'running') status = <span className="status pulse">running the forward pass…</span>;
  else if (engine.kind === 'error') status = <span className="status err">{engine.message}</span>;
  else if (engine.kind === 'ready') status = <span className="status">live · {backend === 'webgpu' ? 'WebGPU' : 'CPU'}{ms ? ` · ${ms.toFixed(0)} ms` : ''}</span>;
  else if (trace?.demo) status = <span className="status">precomputed demo · runs live after a one-time {WEIGHTS_MB} MB download</span>;

  return (
    <div className={`promptbar ${editing ? 'editing' : ''}`}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <input
          ref={input}
          value={text}
          spellCheck={false}
          aria-label="Prompt"
          onFocus={() => setEditing(true)}
          onBlur={() => setTimeout(() => setEditing(false), 150)}
          onChange={(e) => setText(e.target.value)}
          maxLength={240}
        />
        <button type="submit" disabled={busy || !text.trim()}>
          {engine.kind === 'idle' ? 'Run GPT-2' : 'Run'}
        </button>
      </form>
      {editing && (
        <div className="chips" aria-label="Tokens">
          {tokens.map((t, i) => (
            <span key={i} className="chip">
              <span className="t">{show(t.text)}</span>
              <span className="id">{t.id}</span>
            </span>
          ))}
          <span className="count">{tokens.length} tokens{tokens.length > 64 ? ' (first 64 used)' : ''}</span>
        </div>
      )}
      {status}
    </div>
  );
}
