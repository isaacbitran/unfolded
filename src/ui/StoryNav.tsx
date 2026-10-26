// Navigation affordances for the walkthrough: a "scroll to begin" cue, a clickable step rail, and
// previous / next buttons under the text.

import { useEffect, useState } from 'react';
import { story } from '../state/store';
import { STEPS } from '../story/steps';
import { goToStep } from './useScrollStory';

function useStoryPosition() {
  const [s, setS] = useState(0);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      setS((prev) => (Math.abs(prev - story.s) > 0.004 ? story.s : prev));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  return s;
}

const short = (t: string) => t.replace(/^\d+ · /, '');

export function StoryNav() {
  const s = useStoryPosition();
  const cur = Math.max(0, Math.min(STEPS.length - 1, Math.floor(s + 0.06)));
  const hint = Math.max(0, 1 - s / 0.12);
  return (
    <>
      <button className="scroll-hint" style={{ opacity: hint, pointerEvents: hint > 0.2 ? 'auto' : 'none' }} onClick={() => goToStep(1)} aria-label="Start the walkthrough">
        <span>Scroll to begin</span>
        <span className="chev" aria-hidden>
          ↓
        </span>
        <span className="keys">or press → / space</span>
      </button>
      <nav className="rail" aria-label="Steps">
        {STEPS.map((st, i) => {
          const fill = Math.max(0, Math.min(1, s - i));
          return (
            <button key={st.id} className={`dot ${i === cur ? 'on' : ''} ${i < cur ? 'done' : ''}`} onClick={() => goToStep(i)} aria-label={short(st.title)}>
              <span className="tip">{i === 0 ? 'Start' : short(st.title)}</span>
              <span className="pip">
                <span style={{ transform: `scaleY(${i === cur ? fill : i < cur ? 1 : 0})` }} />
              </span>
            </button>
          );
        })}
      </nav>
      <div className="stepnav">
        {cur > 0 && (
          <button onClick={() => goToStep(cur - 1)}>
            ← {cur - 1 === 0 ? 'Start' : short(STEPS[cur - 1].title)}
          </button>
        )}
        {cur < STEPS.length - 1 && (
          <button className="next" onClick={() => goToStep(cur + 1)}>
            {short(STEPS[cur + 1].title)} →
          </button>
        )}
      </div>
    </>
  );
}
