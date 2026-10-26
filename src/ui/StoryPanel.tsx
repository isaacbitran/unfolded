import { useEffect, useMemo, useState } from 'react';
import { STEPS } from '../story/steps';
import { renderProse, tex } from '../story/markup';
import { story } from '../state/store';

/** The text column: title, the learner's question, prose, and the tensor shapes of the current step. */
export function StoryPanel() {
  const [cur, setCur] = useState(0);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const i = Math.max(0, Math.min(STEPS.length - 1, Math.floor(story.s + 0.06)));
      setCur((c) => (c === i ? c : i));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  const html = useMemo(() => STEPS.map((s) => ({ body: renderProse(s.body), shapes: s.shapes.map((x) => tex(x)) })), []);
  return (
    <aside className="panel" aria-live="polite">
      {STEPS.map((s, i) => (
        <section key={s.id} className={`card ${i === cur ? 'on' : i < cur ? 'past' : ''}`} aria-hidden={i !== cur}>
          <div className="step-n">{i === 0 ? 'GPT-2 small · 124M' : `${i} / ${STEPS.length - 1}`}</div>
          <h2>{s.title.replace(/^\d+ · /, '')}</h2>
          <p className="question">{s.question}</p>
          <div className="prose" dangerouslySetInnerHTML={{ __html: html[i].body }} />
          <div className="shapes">
            {html[i].shapes.map((h, j) => (
              <span key={j} className="shape" dangerouslySetInnerHTML={{ __html: h }} />
            ))}
          </div>
        </section>
      ))}
    </aside>
  );
}
