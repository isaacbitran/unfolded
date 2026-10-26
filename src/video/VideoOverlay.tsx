// Film mode (?video): no app UI, a title card, a chapter caption per step, an end card, and
// window.__videoSeek(t) so the renderer can place every frame exactly.

import { useEffect, useRef } from 'react';
import { story } from '../state/store';
import { STEPS } from '../story/steps';
import { tex } from '../story/markup';
import { FILM_SECONDS, cardsAt, storyAt } from './timeline';

declare global {
  interface Window {
    __videoSeek?: (t: number) => Promise<number>;
    __filmSeconds?: number;
  }
}

const smooth = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

export function VideoOverlay() {
  const title = useRef<HTMLDivElement>(null);
  const end = useRef<HTMLDivElement>(null);
  const caps = useRef<(HTMLDivElement | null)[]>([]);
  useEffect(() => {
    story.frozen = true;
    const apply = (t: number) => {
      const s = storyAt(t);
      story.target = story.s = s;
      const c = cardsAt(t);
      if (title.current) title.current.style.opacity = String(c.title);
      if (end.current) end.current.style.opacity = String(c.end);
      caps.current.forEach((el, i) => {
        if (!el) return;
        // caption visible through the body of its step, fading at the edges
        const a = i === 0 ? 0 : smooth((s - i + 0.02) / 0.12) * (1 - smooth((s - i - 0.8) / 0.12)) * (1 - c.end);
        el.style.opacity = String(a);
        el.style.transform = `translateY(${(1 - a) * 8}px)`;
      });
      return s;
    };
    window.__filmSeconds = FILM_SECONDS;
    window.__videoSeek = (t) =>
      new Promise((resolve) => {
        const s = apply(t);
        requestAnimationFrame(() => requestAnimationFrame(() => resolve(s)));
      });
    apply(0);
  }, []);
  return (
    <div className="film">
      <div ref={title} className="film-title">
        <div className="name">Unfolded</div>
        <div className="tag">GPT-2, running in your browser. Every number is real.</div>
      </div>
      {STEPS.map((st, i) => (
        <div key={st.id} ref={(e) => void (caps.current[i] = e)} className="film-cap" style={{ opacity: 0 }}>
          <div className="n">{String(i).padStart(2, '0')}</div>
          <div className="t">{st.title.replace(/^\d+ · /, '')}</div>
          <div className="q">{st.question}</div>
          <div className="sh" dangerouslySetInnerHTML={{ __html: tex(st.shapes[0]) }} />
        </div>
      ))}
      <div ref={end} className="film-end" style={{ opacity: 0 }}>
        <div className="name">Unfolded</div>
        <div className="tag">Type any sentence. Watch GPT-2 compute the answer.</div>
        <div className="url">unfolded-gpt2.vercel.app</div>
      </div>
    </div>
  );
}
